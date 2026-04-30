// ============================================================================
// ESP32 Wi-Fi Provisioning
// ----------------------------------------------------------------------------
// Boots, tries saved creds from NVS. If they fail (or none saved), brings up
// "ESP32-Setup" AP + captive web form. User picks a network, enters password,
// device saves to NVS and reboots into STA mode.
//
// Dependencies (all bundled with the ESP32 Arduino core):
//   - WiFi.h
//   - WebServer.h
//   - Preferences.h
// ============================================================================

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>

// ---------- Configuration ----------------------------------------------------
static const char* AP_SSID          = "ESP32-Setup";
static const char* AP_PASSWORD      = "";         // open AP; set 8+ chars to secure
static const char* NVS_NAMESPACE    = "wifi";
static const char* KEY_SSID         = "ssid";
static const char* KEY_PASS         = "pass";
static const uint32_t CONNECT_TIMEOUT_MS = 15000; // 15s to join saved network
static const uint16_t HTTP_PORT     = 80;

// ---------- Globals ----------------------------------------------------------
WebServer   server(HTTP_PORT);
Preferences prefs;
bool        portalActive = false;

// ---------- Small utilities --------------------------------------------------

// Minimal HTML escape so SSIDs with <, >, &, ", ' don't break the page.
String htmlEscape(const String& in) {
  String out;
  out.reserve(in.length() + 8);
  for (size_t i = 0; i < in.length(); ++i) {
    char c = in[i];
    switch (c) {
      case '&':  out += "&amp;";  break;
      case '<':  out += "&lt;";   break;
      case '>':  out += "&gt;";   break;
      case '"':  out += "&quot;"; break;
      case '\'': out += "&#39;";  break;
      default:   out += c;
    }
  }
  return out;
}

// ---------- NVS helpers ------------------------------------------------------

bool loadCredentials(String& ssid, String& pass) {
  prefs.begin(NVS_NAMESPACE, /*readOnly=*/true);
  ssid = prefs.getString(KEY_SSID, "");
  pass = prefs.getString(KEY_PASS, "");
  prefs.end();
  return ssid.length() > 0;
}

void saveCredentials(const String& ssid, const String& pass) {
  prefs.begin(NVS_NAMESPACE, /*readOnly=*/false);
  prefs.putString(KEY_SSID, ssid);
  prefs.putString(KEY_PASS, pass);
  prefs.end();
}

void clearCredentials() {
  prefs.begin(NVS_NAMESPACE, /*readOnly=*/false);
  prefs.clear();
  prefs.end();
}

// ---------- STA connect ------------------------------------------------------

// Returns true if connected before the timeout elapses.
bool tryConnect(const String& ssid, const String& pass) {
  Serial.printf("[WiFi] Connecting to \"%s\"...\n", ssid.c_str());
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid.c_str(), pass.c_str());

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < CONNECT_TIMEOUT_MS) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WiFi] Connected. IP: %s  RSSI: %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    return true;
  }
  Serial.printf("[WiFi] Connect failed (status=%d).\n", WiFi.status());
  return false;
}

// ---------- HTML rendering ---------------------------------------------------

String renderForm(int netCount) {
  String html;
  html.reserve(2048);
  html += F(
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>ESP32 Wi-Fi Setup</title>"
    "<style>"
    "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:420px;"
    "margin:2em auto;padding:0 1em;color:#222}"
    "h1{font-size:1.3em}"
    "label{display:block;margin:.8em 0 .3em;font-weight:600}"
    "select,input[type=password]{width:100%;padding:.6em;font-size:1em;"
    "box-sizing:border-box;border:1px solid #bbb;border-radius:6px}"
    "button{margin-top:1em;padding:.7em 1em;font-size:1em;border:0;"
    "border-radius:6px;background:#0066cc;color:#fff;cursor:pointer}"
    "button.secondary{background:#666}"
    ".row{display:flex;gap:.5em}.row>*{flex:1}"
    ".hint{color:#666;font-size:.85em;margin-top:.3em}"
    "</style></head><body>"
    "<h1>ESP32 Wi-Fi Setup</h1>"
  );

  html += F("<form method='POST' action='/save'>");
  html += F("<label for='ssid'>Network</label>");
  html += F("<select name='ssid' id='ssid' required>");

  if (netCount <= 0) {
    html += F("<option value=''>(no networks found — try Refresh)</option>");
  } else {
    for (int i = 0; i < netCount; ++i) {
      String ssid = WiFi.SSID(i);
      if (ssid.length() == 0) continue; // filter empty/hidden
      int rssi = WiFi.RSSI(i);
      bool secure = (WiFi.encryptionType(i) != WIFI_AUTH_OPEN);
      String esc = htmlEscape(ssid);
      html += "<option value='" + esc + "'>";
      html += esc;
      html += " (" + String(rssi) + " dBm";
      if (secure) html += ", secured";
      html += ")</option>";
    }
  }
  html += F("</select>");

  html += F("<label for='pass'>Password</label>");
  html += F("<input type='password' name='pass' id='pass' placeholder='leave blank for open networks'>");

  html += F("<div class='row'>");
  html += F("<button type='submit'>Save &amp; Connect</button>");
  html += F("<button type='button' class='secondary' onclick=\"location.href='/refresh'\">Refresh Networks</button>");
  html += F("</div>");
  html += F("<p class='hint'>Device will reboot after saving.</p>");
  html += F("</form>");

  html += F("<p class='hint'><a href='/reset'>Clear saved credentials</a></p>");
  html += F("</body></html>");
  return html;
}

// ---------- HTTP handlers ----------------------------------------------------

void handleRoot() {
  Serial.println("[HTTP] GET /");
  int n = WiFi.scanNetworks(/*async=*/false, /*show_hidden=*/false);
  Serial.printf("[WiFi] Scan found %d networks.\n", n);
  server.send(200, "text/html; charset=utf-8", renderForm(n));
  WiFi.scanDelete();
}

void handleRefresh() {
  Serial.println("[HTTP] GET /refresh");
  // Redirect back to root; root re-scans on load.
  server.sendHeader("Location", "/", true);
  server.send(302, "text/plain", "");
}

void handleSave() {
  Serial.println("[HTTP] POST /save");
  if (!server.hasArg("ssid")) {
    server.send(400, "text/plain", "Missing ssid");
    return;
  }
  String ssid = server.arg("ssid");
  String pass = server.hasArg("pass") ? server.arg("pass") : "";

  if (ssid.length() == 0) {
    server.send(400, "text/plain", "Empty ssid");
    return;
  }

  saveCredentials(ssid, pass);
  Serial.printf("[NVS] Saved creds for \"%s\". Rebooting in 2s...\n", ssid.c_str());

  String body = F(
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<meta http-equiv='refresh' content='8;url=/'>"
    "<title>Saved</title>"
    "<style>body{font-family:sans-serif;max-width:420px;margin:2em auto;padding:0 1em}</style>"
    "</head><body><h1>Credentials saved</h1>"
    "<p>Device is rebooting and will attempt to join <b>");
  body += htmlEscape(ssid);
  body += F("</b>. You can disconnect from <b>ESP32-Setup</b> now.</p></body></html>");

  server.send(200, "text/html; charset=utf-8", body);
  delay(2000);
  ESP.restart();
}

void handleReset() {
  Serial.println("[HTTP] GET /reset");
  clearCredentials();
  server.send(200, "text/html; charset=utf-8",
    F("<!DOCTYPE html><html><body><h1>Credentials cleared</h1>"
      "<p>Rebooting...</p></body></html>"));
  delay(1500);
  ESP.restart();
}

void handleNotFound() {
  server.sendHeader("Location", "/", true);
  server.send(302, "text/plain", "");
}

// ---------- Portal lifecycle -------------------------------------------------

void startPortal() {
  Serial.println("[Portal] Starting AP+STA mode.");
  WiFi.mode(WIFI_AP_STA);

  // STA-side must be disconnected so scanNetworks() works reliably in AP+STA.
  WiFi.disconnect(true, false);

  bool ok = (strlen(AP_PASSWORD) >= 8)
            ? WiFi.softAP(AP_SSID, AP_PASSWORD)
            : WiFi.softAP(AP_SSID);
  if (!ok) {
    Serial.println("[Portal] softAP() failed!");
    return;
  }
  Serial.printf("[Portal] AP \"%s\" up. IP: %s\n",
                AP_SSID, WiFi.softAPIP().toString().c_str());

  server.on("/",        HTTP_GET,  handleRoot);
  server.on("/refresh", HTTP_GET,  handleRefresh);
  server.on("/save",    HTTP_POST, handleSave);
  server.on("/reset",   HTTP_GET,  handleReset);
  server.onNotFound(handleNotFound);
  server.begin();
  portalActive = true;
  Serial.println("[Portal] HTTP server listening on :80");
}

// setup() and loop() live in the main sketch (cam_and_server_code.ino.ino),
// which calls loadCredentials() / tryConnect() / startPortal() from here.
// When portalActive is true, the main loop should route to server.handleClient().
