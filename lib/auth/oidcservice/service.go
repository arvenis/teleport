/*
 * Teleport
 * Copyright (C) 2026  Gravitational, Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// Package oidcservice implements a generic OIDC authentication service for
// Teleport OSS. It handles any standards-compliant OIDC provider (Dex,
// Keycloak, Okta, Auth0, …) by running OIDC discovery and delegating the
// OAuth2 / ID-token flow to the coreos/go-oidc and x/oauth2 libraries.
//
// The implementation mirrors the GitHub SSO flow found in lib/auth/github.go
// and satisfies the auth.OIDCService interface defined in lib/auth/oidc.go,
// which the auth server calls once this service is registered via
// auth.Server.SetOIDCService.
package oidcservice

import (
	"context"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"time"

	gooidc "github.com/coreos/go-oidc/v3/oidc"
	"github.com/gravitational/trace"
	"github.com/jonboulle/clockwork"
	"golang.org/x/oauth2"

	"github.com/gravitational/teleport"
	"github.com/gravitational/teleport/api/constants"
	apidefaults "github.com/gravitational/teleport/api/defaults"
	"github.com/gravitational/teleport/api/types"
	"github.com/gravitational/teleport/lib/auth/authclient"
	"github.com/gravitational/teleport/lib/client/sso"
	"github.com/gravitational/teleport/lib/defaults"
	"github.com/gravitational/teleport/lib/loginrule"
	"github.com/gravitational/teleport/lib/services"
	"github.com/gravitational/teleport/lib/utils"
)

// Backend is the narrow interface that Service requires from the auth server.
// auth.Server satisfies this interface.
type Backend interface {
	// Connector + auth-request storage.
	// Note: these are direct storage calls — they bypass the OIDCService layer
	// to prevent infinite recursion. The adapter in lib/service wires them to
	// auth.Server.Services (the raw backend), not auth.Server itself.
	GetOIDCConnector(ctx context.Context, id string, withSecrets bool) (types.OIDCConnector, error)
	StoreOIDCAuthRequest(ctx context.Context, req types.OIDCAuthRequest, ttl time.Duration) error
	LoadOIDCAuthRequest(ctx context.Context, stateToken string) (*types.OIDCAuthRequest, error)

	// SSO diagnostic info (used by the SSO test flow).
	CreateSSODiagnosticInfo(ctx context.Context, authKind string, authRequestID string, entry types.SSODiagnosticInfo) error

	// Role resolution.
	GetRole(ctx context.Context, name string) (types.Role, error)

	// User management.
	GetUser(ctx context.Context, name string, withSecrets bool) (types.User, error)
	CreateUser(ctx context.Context, user types.User) (types.User, error)
	UpdateUser(ctx context.Context, user types.User) (types.User, error)

	// Login hooks and state.
	CallLoginHooks(ctx context.Context, user types.User) error
	GetUserOrLoginState(ctx context.Context, name string) (services.UserState, error)
	GetLoginRuleEvaluator() loginrule.Evaluator
	ClientOptionsForLogin(userState services.UserState) (authclient.ClientOptions, error)

	// Session / certificate issuance.
	IssueWebSession(ctx context.Context, req IssueWebSessionRequest) (types.WebSession, error)
	IssueSessionCerts(ctx context.Context, req IssueSessionCertsRequest) (sshCert []byte, tlsCert []byte, err error)

	// Cluster metadata.
	GetCertAuthority(ctx context.Context, id types.CertAuthID, withSecrets bool) (types.CertAuthority, error)
	GetClusterName(ctx context.Context) (types.ClusterName, error)
	GetClock() clockwork.Clock
}

// IssueWebSessionRequest carries the parameters needed to create a web session
// after a successful OIDC login.
type IssueWebSessionRequest struct {
	User            string
	Roles           []string
	Traits          map[string][]string
	SessionTTL      time.Duration
	LoginTime       time.Time
	LoginIP         string
	LoginUserAgent  string
	AttestSession   bool
	CreateDeviceToken bool
	Scope           string
}

// IssueSessionCertsRequest carries the parameters needed to issue SSH/TLS
// certificates after a successful OIDC login.
type IssueSessionCertsRequest struct {
	UserState               services.UserState
	SessionTTL              time.Duration
	SSHPubKey               []byte
	TLSPubKey               []byte
	SSHAttestationStatement []byte
	TLSAttestationStatement []byte
	Compatibility           string
	RouteToCluster          string
	KubernetesCluster       string
	LoginIP                 string
	Scope                   string
}

// Config holds the configuration for a Service.
type Config struct {
	// Backend provides access to the auth server and backend storage.
	Backend Backend
	// Logger receives structured log output; defaults to slog.Default().
	Logger *slog.Logger
}

// Service implements auth.OIDCService for standards-compliant OIDC providers.
type Service struct {
	backend Backend
	logger  *slog.Logger
}

// New returns a new Service. The caller is responsible for calling
// auth.Server.SetOIDCService to activate it.
func New(cfg Config) (*Service, error) {
	if cfg.Backend == nil {
		return nil, trace.BadParameter("Backend is required")
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Service{
		backend: cfg.Backend,
		logger:  cfg.Logger.With(teleport.ComponentKey, "oidcservice"),
	}, nil
}

// ─── CreateOIDCAuthRequest ────────────────────────────────────────────────────

// CreateOIDCAuthRequest initiates the OIDC authorization code flow.
// It discovers the provider endpoints, generates an OAuth2 redirect URL with a
// random state token (and optional PKCE challenge), persists the request in
// the backend and returns it with RedirectURL populated.
func (s *Service) CreateOIDCAuthRequest(ctx context.Context, req types.OIDCAuthRequest) (*types.OIDCAuthRequest, error) {
	return s.createOIDCAuthRequestImpl(ctx, req, false)
}

// CreateOIDCAuthRequestForMFA is identical to CreateOIDCAuthRequest but sets
// CheckUser = true so the callback handler verifies the identity against an
// existing authenticated session.
func (s *Service) CreateOIDCAuthRequestForMFA(ctx context.Context, req types.OIDCAuthRequest) (*types.OIDCAuthRequest, error) {
	req.CheckUser = true
	return s.createOIDCAuthRequestImpl(ctx, req, true)
}

func (s *Service) createOIDCAuthRequestImpl(ctx context.Context, req types.OIDCAuthRequest, mfa bool) (*types.OIDCAuthRequest, error) {
	connector, err := s.getOIDCConnector(ctx, req)
	if err != nil {
		return nil, trace.Wrap(err, "loading OIDC connector")
	}

	// Validate client redirect URL for non-web flows.
	if !req.CreateWebSession {
		ceremonyType := sso.CeremonyTypeLogin
		if req.SSOTestFlow {
			ceremonyType = sso.CeremonyTypeTest
		}
		if err := sso.ValidateClientRedirect(req.ClientRedirectURL, ceremonyType, connector.GetClientRedirectSettings()); err != nil {
			return nil, trace.Wrap(err, "invalid client redirect URL")
		}
	}

	// Run OIDC discovery to get provider endpoints.
	provider, err := s.discover(ctx, connector.GetIssuerURL())
	if err != nil {
		return nil, trace.Wrap(err)
	}

	// Pick the redirect URL from the connector that matches ProxyAddress (or
	// the first one if there is no match).
	redirectURL := pickRedirectURL(connector.GetRedirectURLs(), req.ProxyAddress)

	// Generate random state token.
	req.StateToken, err = utils.CryptoRandomHex(defaults.TokenLenBytes)
	if err != nil {
		return nil, trace.Wrap(err)
	}

	// Build OAuth2 config.
	scopes := append([]string{gooidc.ScopeOpenID}, connector.GetScope()...)
	oauth2Config := oauth2.Config{
		ClientID:     connector.GetClientID(),
		ClientSecret: connector.GetClientSecret(),
		RedirectURL:  redirectURL,
		Scopes:       scopes,
		Endpoint:     provider.Endpoint(),
	}

	// Collect extra auth code URL options.
	var opts []oauth2.AuthCodeOption

	if req.LoginHint != "" {
		opts = append(opts, oauth2.SetAuthURLParam("login_hint", req.LoginHint))
	}

	if connector.GetPrompt() != "" {
		opts = append(opts, oauth2.SetAuthURLParam("prompt", connector.GetPrompt()))
	}

	if maxAge, ok := connector.GetMaxAge(); ok {
		opts = append(opts, oauth2.SetAuthURLParam("max_age", fmt.Sprintf("%d", int(maxAge.Seconds()))))
	}

	// PKCE: if the connector has PKCE enabled and the client provided a
	// verifier (CLI flow), derive the challenge and include it.
	if connector.IsPKCEEnabled() && req.PkceVerifier != "" {
		opts = append(opts, oauth2.S256ChallengeOption(req.PkceVerifier))
	}

	req.RedirectURL = oauth2Config.AuthCodeURL(req.StateToken, opts...)

	s.logger.DebugContext(ctx, "Created OIDC auth request",
		"connector", connector.GetName(),
		"redirect_url", req.RedirectURL,
	)

	if err := s.backend.StoreOIDCAuthRequest(ctx, req, defaults.OIDCAuthRequestTTL); err != nil {
		return nil, trace.Wrap(err)
	}
	return &req, nil
}

// ─── ValidateOIDCAuthCallback ─────────────────────────────────────────────────

// ValidateOIDCAuthCallback handles the provider's redirect back to Teleport.
// It exchanges the authorization code for tokens, verifies the ID token,
// maps the claims to Teleport roles, creates/updates the user and issues
// credentials (web session or SSH/TLS certificates).
func (s *Service) ValidateOIDCAuthCallback(ctx context.Context, q url.Values) (*authclient.OIDCAuthResponse, error) {
	event, err := s.validateOIDCAuthCallbackHelper(ctx, q)
	return event, trace.Wrap(err)
}

func (s *Service) validateOIDCAuthCallbackHelper(ctx context.Context, q url.Values) (*authclient.OIDCAuthResponse, error) {
	// IdP may return an error.
	if errParam := q.Get("error"); errParam != "" {
		errDesc := q.Get("error_description")
		oauthErr := trace.OAuth2("invalid_request", errParam, q)
		return nil, trace.WithUserMessage(oauthErr, "OIDC provider returned error: %v [%v]", errDesc, errParam)
	}

	code := q.Get("code")
	if code == "" {
		oauthErr := trace.OAuth2("invalid_request", "missing code parameter", q)
		return nil, trace.WithUserMessage(oauthErr, "Invalid callback parameters from OIDC provider.")
	}

	stateToken := q.Get("state")
	if stateToken == "" {
		oauthErr := trace.OAuth2("invalid_request", "missing state parameter", q)
		return nil, trace.WithUserMessage(oauthErr, "Invalid callback parameters from OIDC provider.")
	}

	// Load the pending request.
	req, err := s.backend.LoadOIDCAuthRequest(ctx, stateToken)
	if err != nil {
		return nil, trace.Wrap(err, "OIDC auth request not found or expired")
	}

	connector, err := s.getOIDCConnector(ctx, *req)
	if err != nil {
		return nil, trace.Wrap(err, "loading OIDC connector")
	}

	// OIDC discovery.
	provider, err := s.discover(ctx, connector.GetIssuerURL())
	if err != nil {
		return nil, trace.Wrap(err)
	}

	// Pick the redirect URL matching the one stored in the request.
	redirectURL := pickRedirectURL(connector.GetRedirectURLs(), req.ProxyAddress)

	oauth2Config := oauth2.Config{
		ClientID:     connector.GetClientID(),
		ClientSecret: connector.GetClientSecret(),
		RedirectURL:  redirectURL,
		Scopes:       append([]string{gooidc.ScopeOpenID}, connector.GetScope()...),
		Endpoint:     provider.Endpoint(),
	}

	// Exchange code → tokens (optionally with PKCE verifier).
	var exchangeOpts []oauth2.AuthCodeOption
	if req.PkceVerifier != "" {
		exchangeOpts = append(exchangeOpts, oauth2.VerifierOption(req.PkceVerifier))
	}

	token, err := oauth2Config.Exchange(ctx, code, exchangeOpts...)
	if err != nil {
		return nil, trace.Wrap(err, "exchanging authorization code for tokens")
	}

	// Extract and verify the ID token.
	rawIDToken, ok := token.Extra("id_token").(string)
	if !ok || rawIDToken == "" {
		return nil, trace.BadParameter("OIDC provider did not return an id_token")
	}

	verifier := provider.Verifier(&gooidc.Config{ClientID: connector.GetClientID()})
	idToken, err := verifier.Verify(ctx, rawIDToken)
	if err != nil {
		return nil, trace.Wrap(err, "verifying OIDC id_token")
	}

	// Extract claims from the ID token.
	var claims map[string]interface{}
	if err := idToken.Claims(&claims); err != nil {
		return nil, trace.Wrap(err, "extracting ID token claims")
	}

	s.logger.DebugContext(ctx, "Extracted OIDC claims",
		"connector", connector.GetName(),
		"subject", idToken.Subject,
	)

	// Determine Teleport username from the configured claim or fall back to sub.
	username, err := extractUsername(connector, claims, idToken.Subject)
	if err != nil {
		return nil, trace.Wrap(err)
	}

	// Convert OIDC claims to the trait map used by Teleport role mapping.
	traits := claimsToTraits(claims, username)

	// Apply claim → role mapping defined in the connector.
	mappings := connector.GetTraitMappings()
	_, roles := services.TraitsToRoles(mappings, traits)
	if len(roles) == 0 {
		return nil, trace.AccessDenied(
			"OIDC claims from %q did not match any roles in connector %q; check claims_to_roles",
			connector.GetIssuerURL(), connector.GetName())
	}

	// Apply login rules (may transform traits/roles).
	evaluationInput := &loginrule.EvaluationInput{Traits: traits}
	evaluationOutput, err := s.backend.GetLoginRuleEvaluator().Evaluate(ctx, evaluationInput)
	if err != nil {
		return nil, trace.Wrap(err, "evaluating login rules")
	}
	traits = evaluationOutput.Traits

	// Re-derive roles from updated traits in case login rules changed them.
	_, roles = services.TraitsToRoles(mappings, traits)
	if len(roles) == 0 {
		return nil, trace.AccessDenied(
			"login rules resulted in empty role set for OIDC user %q", username)
	}

	// Calculate session TTL bounded by role TTL.
	roleSet, err := services.FetchRolesWithContext(roles, s.backend, services.RoleTemplateContext{
		Username: username,
		Traits:   traits,
	})
	if err != nil {
		return nil, trace.Wrap(err, "fetching roles")
	}
	roleTTL := roleSet.AdjustSessionTTL(apidefaults.MaxCertDuration)
	sessionTTL := utils.MinTTL(roleTTL, req.CertTTL)
	if sessionTTL == 0 {
		sessionTTL = apidefaults.MaxCertDuration
	}

	// Diagnostic info for test flows.
	if req.SSOTestFlow {
		if err := s.backend.CreateSSODiagnosticInfo(ctx, types.KindOIDC, req.StateToken, types.SSODiagnosticInfo{
			TestFlow: true,
			Success:  true,
			CreateUserParams: &types.CreateUserParams{
				ConnectorName: connector.GetName(),
				Username:      username,
				Roles:         roles,
				Traits:        traits,
				SessionTTL:    types.Duration(sessionTTL),
			},
			AppliedLoginRules: evaluationOutput.AppliedRules,
		}); err != nil {
			s.logger.WarnContext(ctx, "Failed to write SSO diagnostic info", "error", err)
		}
	}

	// Create or update the Teleport user.
	user, err := s.upsertOIDCUser(ctx, connector, username, idToken.Subject, roles, traits, sessionTTL, req.SSOTestFlow)
	if err != nil {
		return nil, trace.Wrap(err)
	}

	// Test-flow ends here — no certs/sessions are issued.
	if req.SSOTestFlow {
		return &authclient.OIDCAuthResponse{
			Username: username,
			Identity: types.ExternalIdentity{
				ConnectorID: connector.GetName(),
				Username:    idToken.Subject,
			},
			Req: authclient.OIDCAuthRequest{
				ConnectorID:       req.ConnectorID,
				CSRFToken:         req.CSRFToken,
				CreateWebSession:  req.CreateWebSession,
				ClientRedirectURL: req.ClientRedirectURL,
			},
		}, nil
	}

	if err := s.backend.CallLoginHooks(ctx, user); err != nil {
		return nil, trace.Wrap(err)
	}

	userState, err := s.backend.GetUserOrLoginState(ctx, user.GetName())
	if err != nil {
		return nil, trace.Wrap(err, "getting user login state")
	}

	resp := &authclient.OIDCAuthResponse{
		Username: userState.GetName(),
		Identity: types.ExternalIdentity{
			ConnectorID: connector.GetName(),
			Username:    idToken.Subject,
		},
		Req: authclient.OIDCAuthRequest{
			ConnectorID:       req.ConnectorID,
			CSRFToken:         req.CSRFToken,
			CreateWebSession:  req.CreateWebSession,
			ClientRedirectURL: req.ClientRedirectURL,
		},
	}

	// Issue web session (browser flow).
	if req.CreateWebSession {
		session, err := s.backend.IssueWebSession(ctx, IssueWebSessionRequest{
			User:              userState.GetName(),
			Roles:             userState.GetRoles(),
			Traits:            userState.GetTraits(),
			SessionTTL:        sessionTTL,
			LoginTime:         s.backend.GetClock().Now().UTC(),
			LoginIP:           req.ClientLoginIP,
			LoginUserAgent:    req.ClientUserAgent,
			AttestSession:     true,
			CreateDeviceToken: true,
			Scope:             req.Scope,
		})
		if err != nil {
			return nil, trace.Wrap(err, "creating web session")
		}
		resp.Session = session
	}

	// Issue SSH/TLS certificates (CLI flow).
	if len(req.SshPublicKey) != 0 || len(req.TlsPublicKey) != 0 {
		sshCert, tlsCert, err := s.backend.IssueSessionCerts(ctx, IssueSessionCertsRequest{
			UserState:         userState,
			SessionTTL:        sessionTTL,
			SSHPubKey:         req.SshPublicKey,
			TLSPubKey:         req.TlsPublicKey,
			Compatibility:     req.Compatibility,
			RouteToCluster:    req.RouteToCluster,
			KubernetesCluster: req.KubernetesCluster,
			LoginIP:           req.ClientLoginIP,
			Scope:             req.Scope,
		})
		if err != nil {
			return nil, trace.Wrap(err, "issuing session certificates")
		}

		clusterName, err := s.backend.GetClusterName(ctx)
		if err != nil {
			return nil, trace.Wrap(err)
		}

		hostCA, err := s.backend.GetCertAuthority(ctx, types.CertAuthID{
			Type:       types.HostCA,
			DomainName: clusterName.GetClusterName(),
		}, false)
		if err != nil {
			return nil, trace.Wrap(err)
		}

		resp.Cert = sshCert
		resp.TLSCert = tlsCert
		resp.HostSigners = append(resp.HostSigners, hostCA)
	}

	if opts, err := s.backend.ClientOptionsForLogin(userState); err == nil {
		resp.ClientOptions = opts
	} else {
		s.logger.WarnContext(ctx, "Failed to compute client options for OIDC login",
			"user", userState.GetName(), "error", err)
	}

	return resp, nil
}

// ─── helpers ──────────────────────────────────────────────────────────────────

// getOIDCConnector loads the OIDC connector for a request. For test-flow
// requests the connector spec is embedded in the request itself.
func (s *Service) getOIDCConnector(ctx context.Context, req types.OIDCAuthRequest) (types.OIDCConnector, error) {
	if req.SSOTestFlow {
		if req.ConnectorSpec == nil {
			return nil, trace.BadParameter("ConnectorSpec is required for SSO test flow")
		}
		connector, err := types.NewOIDCConnector(req.ConnectorID, *req.ConnectorSpec)
		if err != nil {
			return nil, trace.Wrap(err)
		}
		return connector, nil
	}

	connector, err := s.backend.GetOIDCConnector(ctx, req.ConnectorID, true)
	if err != nil {
		return nil, trace.Wrap(err, "connector %q not found", req.ConnectorID)
	}
	return connector, nil
}

// discover runs OIDC provider discovery (/.well-known/openid-configuration).
func (s *Service) discover(ctx context.Context, issuerURL string) (*gooidc.Provider, error) {
	provider, err := gooidc.NewProvider(ctx, issuerURL)
	if err != nil {
		return nil, trace.Wrap(err, "OIDC discovery failed for issuer %q", issuerURL)
	}
	return provider, nil
}

// pickRedirectURL returns the redirect URL from the connector's list that
// matches the proxy address (host component). Falls back to the first URL.
func pickRedirectURL(redirectURLs []string, proxyAddr string) string {
	if proxyAddr == "" || len(redirectURLs) == 0 {
		if len(redirectURLs) > 0 {
			return redirectURLs[0]
		}
		return ""
	}
	// Strip scheme and path, keeping just the host for comparison.
	proxyHost := strings.ToLower(proxyAddr)
	if u, err := url.Parse(proxyAddr); err == nil && u.Host != "" {
		proxyHost = strings.ToLower(u.Host)
	}
	for _, ru := range redirectURLs {
		u, err := url.Parse(ru)
		if err != nil {
			continue
		}
		if strings.ToLower(u.Host) == proxyHost {
			return ru
		}
	}
	return redirectURLs[0]
}

// extractUsername derives the Teleport username from OIDC claims.
// The connector's user_name_claim takes precedence; if unset we fall back to
// the "email" claim, then the "sub" (subject) from the ID token.
func extractUsername(connector types.OIDCConnector, claims map[string]interface{}, sub string) (string, error) {
	claimKey := connector.GetUsernameClaim()
	if claimKey != "" {
		if v, ok := claims[claimKey]; ok {
			username := fmt.Sprintf("%v", v)
			if username != "" {
				return username, nil
			}
		}
		return "", trace.BadParameter("username claim %q is missing or empty in ID token", claimKey)
	}

	// Prefer email over subject when no explicit claim is configured.
	if email, ok := claims["email"].(string); ok && email != "" {
		return email, nil
	}

	if sub != "" {
		return sub, nil
	}
	return "", trace.BadParameter("could not determine username from OIDC claims")
}

// claimsToTraits converts an ID-token claim map into the trait map that
// Teleport uses for role mapping. Each claim value is normalised to a
// []string so that TraitsToRoles can match against it.
func claimsToTraits(claims map[string]interface{}, username string) map[string][]string {
	traits := make(map[string][]string, len(claims)+1)
	traits[constants.TraitLogins] = []string{username}

	for k, v := range claims {
		switch val := v.(type) {
		case string:
			traits[k] = []string{val}
		case []interface{}:
			strs := make([]string, 0, len(val))
			for _, item := range val {
				strs = append(strs, fmt.Sprintf("%v", item))
			}
			traits[k] = strs
		default:
			traits[k] = []string{fmt.Sprintf("%v", val)}
		}
	}
	return traits
}

// upsertOIDCUser creates a new Teleport user or updates the existing one.
// dryRun = true only instantiates the user in memory (used in SSO test flow).
func (s *Service) upsertOIDCUser(
	ctx context.Context,
	connector types.OIDCConnector,
	username, subject string,
	roles []string,
	traits map[string][]string,
	sessionTTL time.Duration,
	dryRun bool,
) (types.User, error) {
	s.logger.DebugContext(ctx, "Upserting OIDC user",
		"connector", connector.GetName(),
		"username", username,
		"roles", roles,
		"dry_run", dryRun,
	)

	expires := s.backend.GetClock().Now().UTC().Add(sessionTTL)
	user := &types.UserV2{
		Kind:    types.KindUser,
		Version: types.V2,
		Metadata: types.Metadata{
			Name:      username,
			Namespace: apidefaults.Namespace,
			Expires:   &expires,
		},
		Spec: types.UserSpecV2{
			Roles:  roles,
			Traits: traits,
			OIDCIdentities: []types.ExternalIdentity{{
				ConnectorID: connector.GetName(),
				Username:    subject,
			}},
			CreatedBy: types.CreatedBy{
				User: types.UserRef{Name: teleport.UserSystem},
				Time: s.backend.GetClock().Now().UTC(),
				Connector: &types.ConnectorRef{
					Type:     constants.OIDC,
					ID:       connector.GetName(),
					Identity: subject,
				},
			},
		},
	}

	if dryRun {
		return user, nil
	}

	existing, err := s.backend.GetUser(ctx, username, false)
	if err != nil && !trace.IsNotFound(err) {
		return nil, trace.Wrap(err)
	}

	if existing != nil {
		ref := user.GetCreatedBy().Connector
		if !ref.IsSameProvider(existing.GetCreatedBy().Connector) {
			return nil, trace.AlreadyExists(
				"local user %q already exists and was not created by OIDC connector %q",
				username, connector.GetName())
		}
		user.SetRevision(existing.GetRevision())
		if _, err := s.backend.UpdateUser(ctx, user); err != nil {
			return nil, trace.Wrap(err)
		}
	} else {
		if _, err := s.backend.CreateUser(ctx, user); err != nil {
			return nil, trace.Wrap(err)
		}
	}

	return user, nil
}
