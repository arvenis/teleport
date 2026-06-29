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

// Internal test package gives access to unexported helpers (pickRedirectURL,
// extractUsername, claimsToTraits).
package oidcservice

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/go-jose/go-jose/v3"
	josejwt "github.com/go-jose/go-jose/v3/jwt"
	"github.com/gravitational/trace"
	"github.com/jonboulle/clockwork"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/gravitational/teleport/api/constants"
	apidefaults "github.com/gravitational/teleport/api/defaults"
	"github.com/gravitational/teleport/api/types"
	"github.com/gravitational/teleport/lib/auth/authclient"
	"github.com/gravitational/teleport/lib/cryptosuites"
	"github.com/gravitational/teleport/lib/loginrule"
	"github.com/gravitational/teleport/lib/services"
)

// ─── unit tests: unexported helper functions ──────────────────────────────────

func TestPickRedirectURL(t *testing.T) {
	t.Parallel()

	urls := []string{
		"https://proxy1.example.com/v1/webapi/oidc/callback",
		"https://proxy2.example.com/v1/webapi/oidc/callback",
	}

	tests := []struct {
		name      string
		list      []string
		proxyAddr string
		want      string
	}{
		{
			name:      "empty proxy addr returns first",
			list:      urls,
			proxyAddr: "",
			want:      urls[0],
		},
		{
			name:      "matching host returns correct url",
			list:      urls,
			proxyAddr: "proxy2.example.com",
			want:      urls[1],
		},
		{
			name:      "non-matching returns first",
			list:      urls,
			proxyAddr: "unknown.example.com",
			want:      urls[0],
		},
		{
			name:      "proxy addr with scheme",
			list:      urls,
			proxyAddr: "https://proxy1.example.com",
			want:      urls[0],
		},
		{
			name:      "empty list returns empty string",
			list:      nil,
			proxyAddr: "proxy1.example.com",
			want:      "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := pickRedirectURL(tc.list, tc.proxyAddr)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestExtractUsername(t *testing.T) {
	t.Parallel()

	makeConnector := func(usernameClaim string) types.OIDCConnector {
		spec := types.OIDCConnectorSpecV3{
			ClientID:     "teleport",
			ClientSecret: "secret",
			RedirectURLs: []string{"https://example.com/callback"},
			ClaimsToRoles: []types.ClaimMapping{
				{Claim: "groups", Value: "admins", Roles: []string{"admin"}},
			},
			UsernameClaim: usernameClaim,
		}
		c, err := types.NewOIDCConnector("test", spec)
		require.NoError(t, err)
		return c
	}

	tests := []struct {
		name      string
		connector types.OIDCConnector
		claims    map[string]interface{}
		sub       string
		wantUser  string
		wantErr   bool
	}{
		{
			name:      "explicit username_claim present",
			connector: makeConnector("preferred_username"),
			claims:    map[string]interface{}{"preferred_username": "alice"},
			sub:       "12345",
			wantUser:  "alice",
		},
		{
			name:      "falls back to email claim",
			connector: makeConnector(""),
			claims:    map[string]interface{}{"email": "alice@example.com"},
			sub:       "12345",
			wantUser:  "alice@example.com",
		},
		{
			name:      "falls back to sub when no email",
			connector: makeConnector(""),
			claims:    map[string]interface{}{"name": "Alice"},
			sub:       "sub-abc-123",
			wantUser:  "sub-abc-123",
		},
		{
			name:      "missing explicit claim returns error",
			connector: makeConnector("missing_claim"),
			claims:    map[string]interface{}{"email": "alice@example.com"},
			sub:       "12345",
			wantErr:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := extractUsername(tc.connector, tc.claims, tc.sub)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.wantUser, got)
		})
	}
}

func TestClaimsToTraits(t *testing.T) {
	t.Parallel()

	claims := map[string]interface{}{
		"email":  "alice@example.com",
		"groups": []interface{}{"admins", "developers"},
		"sub":    "12345",
		"count":  float64(42),
	}

	traits := claimsToTraits(claims, "alice@example.com")

	assert.Equal(t, []string{"alice@example.com"}, traits[constants.TraitLogins])
	assert.Equal(t, []string{"alice@example.com"}, traits["email"])
	assert.Equal(t, []string{"admins", "developers"}, traits["groups"])
	assert.Equal(t, []string{"12345"}, traits["sub"])
	assert.Equal(t, []string{"42"}, traits["count"])
}

// ─── fake OIDC provider ───────────────────────────────────────────────────────

// fakeOIDCProvider is a test HTTP server acting as a minimal OIDC provider:
// serves OpenID discovery, JWKS, and token endpoint.
type fakeOIDCProvider struct {
	server      *httptest.Server
	signer      jose.Signer
	jwks        *jose.JSONWebKeySet
	clock       clockwork.Clock
	mu          sync.Mutex
	codeToToken map[string][]byte
}

func newFakeOIDCProvider(t *testing.T) *fakeOIDCProvider {
	t.Helper()

	key, err := cryptosuites.GenerateKeyWithAlgorithm(cryptosuites.ECDSAP256)
	require.NoError(t, err)

	const keyID = "test-key"
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: key},
		(&jose.SignerOptions{}).WithType("JWT").WithHeader("kid", keyID),
	)
	require.NoError(t, err)

	jwks := &jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       key.Public(),
		Use:       "sig",
		Algorithm: string(jose.ES256),
		KeyID:     keyID,
	}}}

	p := &fakeOIDCProvider{
		signer:      signer,
		jwks:        jwks,
		clock:       clockwork.NewRealClock(),
		codeToToken: make(map[string][]byte),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", p.handleDiscovery)
	mux.HandleFunc("/keys", p.handleJWKS)
	mux.HandleFunc("/token", p.handleToken)
	p.server = httptest.NewServer(mux)
	t.Cleanup(p.server.Close)
	return p
}

func (p *fakeOIDCProvider) issuerURL() string { return p.server.URL }

func (p *fakeOIDCProvider) handleDiscovery(w http.ResponseWriter, _ *http.Request) {
	doc := map[string]any{
		"issuer":                                p.issuerURL(),
		"authorization_endpoint":                p.issuerURL() + "/auth",
		"token_endpoint":                        p.issuerURL() + "/token",
		"jwks_uri":                              p.issuerURL() + "/keys",
		"response_types_supported":              []string{"code"},
		"subject_types_supported":               []string{"public"},
		"id_token_signing_alg_values_supported": []string{"ES256"},
		"scopes_supported":                      []string{"openid", "email", "profile", "groups"},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(doc)
}

func (p *fakeOIDCProvider) handleJWKS(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(p.jwks)
}

func (p *fakeOIDCProvider) handleToken(w http.ResponseWriter, r *http.Request) {
	_ = r.ParseForm()
	code := r.FormValue("code")
	p.mu.Lock()
	resp, ok := p.codeToToken[code]
	p.mu.Unlock()
	if !ok {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_grant"}`))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(resp)
}

// issueIDToken registers an auth code that the token endpoint will exchange for
// a signed ID token containing the given extra claims.
func (p *fakeOIDCProvider) issueIDToken(t *testing.T, code, clientID, sub string, extraClaims map[string]interface{}) {
	t.Helper()

	now := p.clock.Now()
	stdClaims := josejwt.Claims{
		Issuer:    p.issuerURL(),
		Subject:   sub,
		Audience:  josejwt.Audience{clientID},
		IssuedAt:  josejwt.NewNumericDate(now),
		NotBefore: josejwt.NewNumericDate(now),
		Expiry:    josejwt.NewNumericDate(now.Add(10 * time.Minute)),
	}

	allClaims := map[string]interface{}{
		"iss": stdClaims.Issuer,
		"sub": stdClaims.Subject,
		"aud": clientID,
		"iat": stdClaims.IssuedAt,
		"nbf": stdClaims.NotBefore,
		"exp": stdClaims.Expiry,
	}
	for k, v := range extraClaims {
		allClaims[k] = v
	}

	rawJWT, err := josejwt.Signed(p.signer).Claims(allClaims).CompactSerialize()
	require.NoError(t, err)

	tokenResp, err := json.Marshal(map[string]interface{}{
		"access_token": "fake-access-token",
		"token_type":   "Bearer",
		"expires_in":   3600,
		"id_token":     rawJWT,
	})
	require.NoError(t, err)

	p.mu.Lock()
	p.codeToToken[code] = tokenResp
	p.mu.Unlock()
}

// ─── fake Backend ─────────────────────────────────────────────────────────────

type testBackend struct {
	mu           sync.Mutex
	authRequests map[string]*types.OIDCAuthRequest
	users        map[string]*types.UserV2
	clock        clockwork.Clock
	connector    types.OIDCConnector
}

func newTestBackend(clock clockwork.Clock, connector types.OIDCConnector) *testBackend {
	return &testBackend{
		authRequests: make(map[string]*types.OIDCAuthRequest),
		users:        make(map[string]*types.UserV2),
		clock:        clock,
		connector:    connector,
	}
}

func (b *testBackend) GetOIDCConnector(_ context.Context, _ string, _ bool) (types.OIDCConnector, error) {
	return b.connector, nil
}

func (b *testBackend) StoreOIDCAuthRequest(_ context.Context, req types.OIDCAuthRequest, _ time.Duration) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	copy := req
	b.authRequests[req.StateToken] = &copy
	return nil
}

func (b *testBackend) LoadOIDCAuthRequest(_ context.Context, state string) (*types.OIDCAuthRequest, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	req, ok := b.authRequests[state]
	if !ok {
		return nil, trace.NotFound("auth request %q not found", state)
	}
	return req, nil
}

func (b *testBackend) CreateSSODiagnosticInfo(_ context.Context, _, _ string, _ types.SSODiagnosticInfo) error {
	return nil
}

func (b *testBackend) GetRole(_ context.Context, name string) (types.Role, error) {
	role, err := types.NewRole(name, types.RoleSpecV6{})
	return role, trace.Wrap(err)
}

func (b *testBackend) GetUser(_ context.Context, name string, _ bool) (types.User, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	u, ok := b.users[name]
	if !ok {
		return nil, trace.NotFound("user %q not found", name)
	}
	return u, nil
}

func (b *testBackend) CreateUser(_ context.Context, user types.User) (types.User, error) {
	v2, ok := user.(*types.UserV2)
	if !ok {
		return nil, trace.BadParameter("expected *types.UserV2, got %T", user)
	}
	b.mu.Lock()
	b.users[user.GetName()] = v2
	b.mu.Unlock()
	return user, nil
}

func (b *testBackend) UpdateUser(_ context.Context, user types.User) (types.User, error) {
	v2, ok := user.(*types.UserV2)
	if !ok {
		return nil, trace.BadParameter("expected *types.UserV2, got %T", user)
	}
	b.mu.Lock()
	b.users[user.GetName()] = v2
	b.mu.Unlock()
	return user, nil
}

func (b *testBackend) CallLoginHooks(_ context.Context, _ types.User) error { return nil }

func (b *testBackend) GetUserOrLoginState(_ context.Context, name string) (services.UserState, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	u, ok := b.users[name]
	if !ok {
		return nil, trace.NotFound("user %q not found", name)
	}
	return u, nil
}

func (b *testBackend) GetLoginRuleEvaluator() loginrule.Evaluator {
	return loginrule.NullEvaluator{}
}

func (b *testBackend) ClientOptionsForLogin(_ services.UserState) (authclient.ClientOptions, error) {
	return authclient.ClientOptions{}, nil
}

func (b *testBackend) IssueWebSession(_ context.Context, req IssueWebSessionRequest) (types.WebSession, error) {
	ttl := req.SessionTTL
	if ttl == 0 {
		ttl = apidefaults.CertDuration
	}
	s, err := types.NewWebSession(
		fmt.Sprintf("session-%s", req.User),
		types.KindWebSession,
		types.WebSessionSpecV2{
			User:    req.User,
			Expires: b.clock.Now().Add(ttl),
		},
	)
	return s, trace.Wrap(err)
}

func (b *testBackend) IssueSessionCerts(_ context.Context, _ IssueSessionCertsRequest) ([]byte, []byte, error) {
	return []byte("fake-ssh-cert"), []byte("fake-tls-cert"), nil
}

func (b *testBackend) GetCertAuthority(_ context.Context, _ types.CertAuthID, _ bool) (types.CertAuthority, error) {
	ca, err := types.NewCertAuthority(types.CertAuthoritySpecV2{
		Type:        types.HostCA,
		ClusterName: "test-cluster",
	})
	return ca, trace.Wrap(err)
}

func (b *testBackend) GetClusterName(_ context.Context) (types.ClusterName, error) {
	cn, err := types.NewClusterName(types.ClusterNameSpecV2{
		ClusterName: "test-cluster",
		ClusterID:   "test-cluster-id",
	})
	return cn, trace.Wrap(err)
}

func (b *testBackend) GetClock() clockwork.Clock { return b.clock }

// ─── test connector factory ───────────────────────────────────────────────────

func buildTestConnector(t *testing.T, issuerURL string) types.OIDCConnector {
	t.Helper()
	spec := types.OIDCConnectorSpecV3{
		IssuerURL:    issuerURL,
		ClientID:     "teleport",
		ClientSecret: "secret",
		RedirectURLs: []string{"https://proxy.example.com/v1/webapi/oidc/callback"},
		Scope:        []string{"email", "groups"},
		ClaimsToRoles: []types.ClaimMapping{
			{Claim: "groups", Value: "admins", Roles: []string{"admin"}},
		},
	}
	c, err := types.NewOIDCConnector("test-connector", spec)
	require.NoError(t, err)
	return c
}

// ─── integration tests ────────────────────────────────────────────────────────

func TestCreateOIDCAuthRequest(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	result, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:       "test-connector",
		CreateWebSession:  true,
		ClientRedirectURL: "/web",
	})
	require.NoError(t, err)

	require.NotEmpty(t, result.StateToken)
	require.NotEmpty(t, result.RedirectURL)

	redirectURL, err := url.Parse(result.RedirectURL)
	require.NoError(t, err)

	// Redirect must point to the fake provider's auth endpoint.
	assert.Equal(t, provider.issuerURL()+"/auth",
		redirectURL.Scheme+"://"+redirectURL.Host+redirectURL.Path)

	// State token is echoed in the redirect query.
	assert.Equal(t, result.StateToken, redirectURL.Query().Get("state"))

	// Scopes include openid plus connector extras.
	scope := redirectURL.Query().Get("scope")
	assert.Contains(t, scope, "openid")
	assert.Contains(t, scope, "email")
	assert.Contains(t, scope, "groups")

	// Request is persisted in the backend.
	stored, err := backend.LoadOIDCAuthRequest(context.Background(), result.StateToken)
	require.NoError(t, err)
	assert.Equal(t, result.StateToken, stored.StateToken)
}

func TestValidateOIDCAuthCallback_WebSession(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	// Initiate the browser login flow.
	authReq, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:       "test-connector",
		CreateWebSession:  true,
		ClientRedirectURL: "/web",
	})
	require.NoError(t, err)

	// Fake provider issues an ID token for alice with admin group membership.
	const code = "auth-code-web"
	provider.issueIDToken(t, code, "teleport", "sub-alice", map[string]interface{}{
		"email":  "alice@example.com",
		"groups": []string{"admins"},
	})

	resp, err := svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"code":  []string{code},
		"state": []string{authReq.StateToken},
	})
	require.NoError(t, err)

	assert.Equal(t, "alice@example.com", resp.Username)
	assert.Equal(t, "test-connector", resp.Identity.ConnectorID)
	assert.Equal(t, "sub-alice", resp.Identity.Username)
	require.NotNil(t, resp.Session, "web session must be set")
	assert.Equal(t, "alice@example.com", resp.Session.GetUser())

	// User must be persisted with the mapped role.
	user, err := backend.GetUser(context.Background(), "alice@example.com", false)
	require.NoError(t, err)
	assert.Contains(t, user.GetRoles(), "admin")

	// Web session flow must not issue certs.
	assert.Empty(t, resp.Cert)
	assert.Empty(t, resp.TLSCert)
}

func TestValidateOIDCAuthCallback_CLICerts(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	// CLI login includes public keys.
	authReq, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:      "test-connector",
		CreateWebSession: false,
		SshPublicKey:     []byte("fake-ssh-pubkey"),
		TlsPublicKey:     []byte("fake-tls-pubkey"),
		// tsh listener redirect: must have path=/callback and ?secret_key=.
		ClientRedirectURL: "http://127.0.0.1:58541/callback?secret_key=test",
	})
	require.NoError(t, err)

	const code = "auth-code-cli"
	provider.issueIDToken(t, code, "teleport", "sub-bob", map[string]interface{}{
		"email":  "bob@example.com",
		"groups": []string{"admins"},
	})

	resp, err := svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"code":  []string{code},
		"state": []string{authReq.StateToken},
	})
	require.NoError(t, err)

	assert.Equal(t, "bob@example.com", resp.Username)
	// CLI flow must return certs, not a web session.
	assert.Equal(t, []byte("fake-ssh-cert"), resp.Cert)
	assert.Equal(t, []byte("fake-tls-cert"), resp.TLSCert)
	assert.Nil(t, resp.Session)
}

func TestValidateOIDCAuthCallback_UnmappedClaims(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	authReq, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:       "test-connector",
		CreateWebSession:  true,
		ClientRedirectURL: "/web",
	})
	require.NoError(t, err)

	// User is in "users" group only — no role mapping exists for it.
	provider.issueIDToken(t, "unmapped-code", "teleport", "sub-nobody", map[string]interface{}{
		"email":  "nobody@example.com",
		"groups": []string{"users"},
	})

	_, err = svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"code":  []string{"unmapped-code"},
		"state": []string{authReq.StateToken},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "did not match any roles")
}

func TestValidateOIDCAuthCallback_ProviderError(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	// Simulate the IdP returning an error instead of an authorization code.
	_, err = svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"error":             []string{"access_denied"},
		"error_description": []string{"User cancelled."},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "access_denied")
}

func TestValidateOIDCAuthCallback_SSOTestFlow(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	connSpec := connector.(*types.OIDCConnectorV3).Spec

	// SSOTestFlow embeds the connector spec inline rather than loading from backend.
	authReq, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:      "test-connector",
		SSOTestFlow:      true,
		ConnectorSpec:    &connSpec,
		CreateWebSession: true,
	})
	require.NoError(t, err)

	const code = "test-flow-code"
	provider.issueIDToken(t, code, "teleport", "sub-tester", map[string]interface{}{
		"email":  "tester@example.com",
		"groups": []string{"admins"},
	})

	resp, err := svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"code":  []string{code},
		"state": []string{authReq.StateToken},
	})
	require.NoError(t, err)

	// Test flow returns identity but must NOT issue a session or certs.
	assert.Equal(t, "tester@example.com", resp.Username)
	assert.Nil(t, resp.Session, "SSO test flow must not issue a web session")
	assert.Empty(t, resp.Cert)
	assert.Empty(t, resp.TLSCert)
}

func TestValidateOIDCAuthCallback_UserExistsFromDifferentConnector(t *testing.T) {
	t.Parallel()

	provider := newFakeOIDCProvider(t)
	connector := buildTestConnector(t, provider.issuerURL())
	backend := newTestBackend(clockwork.NewRealClock(), connector)

	// Pre-create a user that was originally created by a different connector.
	existing := &types.UserV2{
		Kind:    types.KindUser,
		Version: types.V2,
		Metadata: types.Metadata{
			Name:      "alice@example.com",
			Namespace: apidefaults.Namespace,
		},
		Spec: types.UserSpecV2{
			CreatedBy: types.CreatedBy{
				Connector: &types.ConnectorRef{
					Type: constants.OIDC,
					ID:   "different-connector",
				},
			},
		},
	}
	backend.mu.Lock()
	backend.users["alice@example.com"] = existing
	backend.mu.Unlock()

	svc, err := New(Config{Backend: backend})
	require.NoError(t, err)

	authReq, err := svc.CreateOIDCAuthRequest(context.Background(), types.OIDCAuthRequest{
		ConnectorID:       "test-connector",
		CreateWebSession:  true,
		ClientRedirectURL: "/web",
	})
	require.NoError(t, err)

	provider.issueIDToken(t, "conflict-code", "teleport", "sub-alice", map[string]interface{}{
		"email":  "alice@example.com",
		"groups": []string{"admins"},
	})

	_, err = svc.ValidateOIDCAuthCallback(context.Background(), url.Values{
		"code":  []string{"conflict-code"},
		"state": []string{authReq.StateToken},
	})
	require.Error(t, err)
	assert.True(t, trace.IsAlreadyExists(err), "expected AlreadyExists error, got: %v", err)
}
