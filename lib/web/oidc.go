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

package web

import (
	"net"
	"net/http"
	"strings"

	"github.com/gravitational/trace"
	"github.com/julienschmidt/httprouter"
	"golang.org/x/oauth2"

	"github.com/gravitational/teleport/api/types"
	"github.com/gravitational/teleport/lib/auth"
	"github.com/gravitational/teleport/lib/client"
	"github.com/gravitational/teleport/lib/client/sso"
	"github.com/gravitational/teleport/lib/httplib"
)

// oidcLoginWeb handles browser-initiated OIDC login.
// It creates an OIDCAuthRequest that asks for a web session and redirects the
// browser to the OIDC provider's authorization endpoint.
func (h *Handler) oidcLoginWeb(w http.ResponseWriter, r *http.Request, p httprouter.Params) string {
	logger := h.logger.With("auth", "oidc")
	logger.DebugContext(r.Context(), "OIDC web login start")

	req, err := ParseSSORequestParams(r)
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to extract SSO parameters from request", "error", err)
		return sso.LoginFailedRedirectURL
	}

	remoteAddr, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to parse remote address", "error", err)
		return sso.LoginFailedRedirectURL
	}

	response, err := h.cfg.ProxyClient.CreateOIDCAuthRequest(r.Context(), types.OIDCAuthRequest{
		CSRFToken:         req.CSRFToken,
		ConnectorID:       req.ConnectorID,
		CreateWebSession:  true,
		ClientRedirectURL: req.ClientRedirectURL,
		ClientLoginIP:     remoteAddr,
		ClientUserAgent:   r.UserAgent(),
		ProxyAddress:      r.Host,
	})
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to create OIDC auth request", "error", err)
		return sso.LoginFailedRedirectURL
	}

	return response.RedirectURL
}

// oidcLoginConsole handles tsh-initiated (CLI / console) OIDC login.
// The client provides its public keys and a redirect URL; we create an
// OIDCAuthRequest that will issue certificates rather than a web session.
func (h *Handler) oidcLoginConsole(w http.ResponseWriter, r *http.Request, p httprouter.Params) (any, error) {
	logger := h.logger.With("auth", "oidc")
	logger.DebugContext(r.Context(), "OIDC console login start")

	req := new(client.SSOLoginConsoleReq)
	if err := httplib.ReadResourceJSON(r, req); err != nil {
		logger.ErrorContext(r.Context(), "Error reading request JSON", "error", err)
		return nil, trace.AccessDenied("%s", SSOLoginFailureMessage)
	}
	if err := req.CheckAndSetDefaults(); err != nil {
		logger.ErrorContext(r.Context(), "Missing required request parameters", "error", err)
		return nil, trace.AccessDenied("%s", SSOLoginFailureMessage)
	}

	remoteAddr, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to parse remote address", "error", err)
		return nil, trace.AccessDenied("%s", SSOLoginFailureMessage)
	}

	// If the client did not provide a PKCE verifier, generate one.  This keeps
	// the server in control when the client is an older tsh that predates PKCE.
	pkceVerifier := req.PKCEVerifier
	if pkceVerifier == "" {
		pkceVerifier = oauth2.GenerateVerifier()
	}

	response, err := h.cfg.ProxyClient.CreateOIDCAuthRequest(r.Context(), types.OIDCAuthRequest{
		ConnectorID:             req.ConnectorID,
		SshPublicKey:            req.SSHPubKey,
		TlsPublicKey:            req.TLSPubKey,
		SshAttestationStatement: req.SSHAttestationStatement.ToProto(),
		TlsAttestationStatement: req.TLSAttestationStatement.ToProto(),
		CertTTL:                 req.CertTTL,
		ClientRedirectURL:       req.RedirectURL,
		Compatibility:           req.Compatibility,
		RouteToCluster:          req.RouteToCluster,
		KubernetesCluster:       req.KubernetesCluster,
		ClientLoginIP:           remoteAddr,
		PkceVerifier:            pkceVerifier,
		ProxyAddress:            r.Host,
		Scope:                   req.Scope,
	})
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to create OIDC auth request", "error", err)
		if strings.Contains(err.Error(), auth.InvalidClientRedirectErrorMessage) {
			return nil, trace.AccessDenied("%s", SSOLoginFailureInvalidRedirect)
		}
		return nil, trace.AccessDenied("%s", SSOLoginFailureMessage)
	}

	return &client.SSOLoginConsoleResponse{RedirectURL: response.RedirectURL}, nil
}

// oidcCallback processes the OIDC provider's authorization callback.
// It exchanges the authorization code for tokens, verifies the identity and
// either sets a web-session cookie or returns credentials to the tsh client.
func (h *Handler) oidcCallback(w http.ResponseWriter, r *http.Request, p httprouter.Params) string {
	logger := h.logger.With("auth", "oidc")
	logger.DebugContext(r.Context(), "OIDC callback start", "query", r.URL.Query())

	response, err := h.cfg.ProxyClient.ValidateOIDCAuthCallback(r.Context(), r.URL.Query())
	if err != nil {
		logger.ErrorContext(r.Context(), "Error processing OIDC callback", "error", err)

		// Try to find the original request so we can redirect with a useful error.
		if stateToken := r.URL.Query().Get("state"); stateToken != "" {
			if req, getErr := h.cfg.ProxyClient.GetOIDCAuthRequest(r.Context(), stateToken); getErr == nil && !req.CreateWebSession {
				if redURL, encErr := RedirectURLWithError(req.ClientRedirectURL, err); encErr == nil {
					return redURL.String()
				}
			}
		}
		return sso.LoginFailedBadCallbackRedirectURL
	}

	// Browser (web session) flow: set the session cookie and redirect.
	if response.Req.CreateWebSession {
		logger.InfoContext(r.Context(), "Redirecting to web browser after OIDC login")

		res := &SSOCallbackResponse{
			CSRFToken:         response.Req.CSRFToken,
			Username:          response.Username,
			SessionName:       response.Session.GetName(),
			SessionExpiry:     response.Session.Expiry(),
			ClientRedirectURL: response.Req.ClientRedirectURL,
		}

		if err := SSOSetWebSessionAndRedirectURL(w, r, res, true); err != nil {
			logger.ErrorContext(r.Context(), "Failed to set web session", "error", err)
			return sso.LoginFailedRedirectURL
		}

		if dwt := response.Session.GetDeviceWebToken(); dwt != nil {
			logger.DebugContext(r.Context(), "OIDC web session created with device web token")
			redirectPath, err := BuildDeviceWebRedirectPath(dwt, res.ClientRedirectURL)
			if err != nil {
				logger.DebugContext(r.Context(), "Invalid device web token", "error", err)
			}
			return redirectPath
		}
		return res.ClientRedirectURL
	}

	// CLI (console) flow: encode certificates into a redirect to the tsh listener.
	logger.InfoContext(r.Context(), "Redirecting to console login after OIDC callback")
	if len(response.Req.SSHPubKey)+len(response.Req.TLSPubKey) == 0 {
		logger.ErrorContext(r.Context(), "OIDC callback: neither web session nor public key provided")
		return sso.LoginFailedRedirectURL
	}

	redirectURL, err := ConstructSSHResponse(AuthParams{
		ClientRedirectURL: response.Req.ClientRedirectURL,
		Username:          response.Username,
		Identity:          response.Identity,
		Session:           response.Session,
		Cert:              response.Cert,
		TLSCert:           response.TLSCert,
		HostSigners:       response.HostSigners,
		FIPS:              h.cfg.FIPS,
		ClientOptions:     response.ClientOptions,
	})
	if err != nil {
		logger.ErrorContext(r.Context(), "Failed to construct SSH response for OIDC callback", "error", err)
		return sso.LoginFailedRedirectURL
	}

	return redirectURL.String()
}

