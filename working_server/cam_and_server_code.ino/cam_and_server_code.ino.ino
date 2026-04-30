// ===========================
// Select camera model in board_config.h

// ===========================
#include "board_config.h"

#include "esp_camera.h"
#include "WiFi.h"
#include <WebServer.h>

#include <HTTPClient.h>
const char* pythonServer = "http://192.168.0.152:5000/trigger";  //

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

// ===========================
// Wi-Fi credentials (legacy — now provisioned via wifi_provisioning.ino / NVS)
// ===========================
// const char *ssid =  "Jio Ant";  //"T-Mobile Hotspot_6218_2.4GHz";   //"Jio Ant";
// const char *password = "sigmaalpha";  //"54786218"; //"sigmaalpha";
//192.168.0.31 = jioant
//192.168.0.211 = TMOBILE

const char* BEACON_MAC = "dd:34:02:0a:2d:f1"; 
const char* BEACON_MAC2 = "";//"dd:34:02:0a:2d:da"; 
const int RSSI_THRESHOLD   = -70;   // Stronger = closer (turn ON)
// const int RSSI_NEAR = -74;  
// const int RSSI_FAR = -74;
volatile int scans = 0;

volatile int lastBeaconRSSI = -100;
volatile unsigned long lastBeaconTime = 0;
unsigned long beaconTimeoutMs = 3500; // 3.5 seconds without detection = beacon lost


bool cameraActive = false;
bool flagStopScan = false;

volatile bool beaconDetectedNear = false;
volatile bool beaconDetectedFar = false;
volatile bool beaconFound = false;
// RSSI-based state, independent of cameraActive. cameraActive can desync from
// the backend (e.g. failed near-POST, ESP32 reset), so we drive near/far purely
// off whether RSSI is currently strong enough.
volatile bool beaconCurrentlyNear = false;

static unsigned long blockUntil = 0;       
// int scanTime = 0.8f; //In seconds
BLEScan* pBLEScan;

void startCameraServer();
void setupLedFlash();
bool startCamera();
bool stopCamera();

// Defined in wifi_provisioning.ino
bool loadCredentials(String& ssid, String& pass);
bool tryConnect(const String& ssid, const String& pass);
void startPortal();
extern bool portalActive;
extern WebServer server;

class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {
    void onResult(BLEAdvertisedDevice advertisedDevice) {
      BLEAddress addr = advertisedDevice.getAddress();
      const uint8_t* bytes = addr.getNative();  // raw MAC bytes
    
      // Compare directly as bytes
      bool match =
          bytes[5] == 0xdd &&
          bytes[4] == 0x34 &&
          bytes[3] == 0x02 &&
          bytes[2] == 0x0a &&
          bytes[1] == 0x2d &&
          bytes[0] == 0xf1;


      if (match) {
          int rssi = advertisedDevice.getRSSI();
          lastBeaconRSSI = rssi;
          lastBeaconTime = millis();
          //if (mac == BEACON_MAC) {  //|| mac == BEACON_MAC2) {
          //Serial.printf("KBeacon found! RSSI: %d dBm\n", rssi);
          

          // Edge-triggered: fire on transition between near and far, irrespective
          // of cameraActive. JUST SET FLAGS - NO BLOCKING OPERATIONS.
          bool nearNow = (rssi >= RSSI_THRESHOLD);
          if (nearNow && !beaconCurrentlyNear) {
            beaconCurrentlyNear = true;
            beaconDetectedNear  = true;
          } else if (!nearNow && beaconCurrentlyNear) {
            beaconCurrentlyNear = false;
            beaconDetectedFar   = true;
          }
          // if (!cameraActive && rssi >= RSSI_NEAR) { 
          //   // Signal is STRONGER (less negative) than threshold = beacon is close
            
          //   Serial.println("Beacon is CLOSE (strong signal)");
          //   cameraActive = startCamera();
          //   flagStopScan = true;
          //   blockUntil = millis() + 5000;
            
          // } 
          // // camera on but beacon far -> turn off camera
          // else if (cameraActive && rssi < RSSI_FAR) { 
          //   // Signal is WEAKER (more negative) than threshold = beacon is far
          //   Serial.println("Beacon is FAR (weak signal)");
          //   cameraActive = !stopCamera();
          
          // }
        }
        
      scans += 1;
    }
    
             
};

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  //Check PSRAM FIRST
  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found! Camera will not work properly.");
    Serial.println("Check Tools > PSRAM setting in Arduino IDE");
    while(1) delay(1000); // Stop here
  }

  //add heap monitoring
  Serial.printf("Free Heap before init: %d\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; //changed from 2
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 3;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 12;
      config.fb_count = 3;
      //config.grab_mode = CAMERA_GRAB_LATEST;
      config.frame_size = FRAMESIZE_QVGA;//FRAMESIZE_240X240;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  // ---- Wi-Fi: try NVS-saved creds, fall back to provisioning portal ----
  String savedSsid, savedPass;
  bool connected = false;
  if (loadCredentials(savedSsid, savedPass)) {
    Serial.printf("[NVS] Loaded saved SSID: \"%s\"\n", savedSsid.c_str());
    connected = tryConnect(savedSsid, savedPass);
    if (!connected) {
      Serial.println("[WiFi] Saved credentials failed; starting portal.");
    }
  } else {
    Serial.println("[NVS] No saved credentials; starting portal.");
  }

  if (!connected) {
    startPortal();
    return; // stay in portal mode; loop() will service HTTP requests
  }

  WiFi.setSleep(false);

  startCameraServer();

  Serial.println("Scanning...");

  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan(); //create new scan
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setActiveScan(true); //active scan uses more power, but get results faster
  pBLEScan->setInterval(80);
  pBLEScan->setWindow(60);  // less or equal setInterval value
  pBLEScan->start(0, nullptr); // continuous, nonblocking scan


  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

// ===========================
// Camera control stubs
// ===========================
bool startCamera() {
  Serial.println("Beacon nearby → notifying Python");
  HTTPClient http;
  http.begin(pythonServer);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST("{\"event\":\"beacon_near\"}");   // send POST request
   // STOP BLE SCANNING for delay time
 

  Serial.printf("HTTP Response code: %d\n", httpCode);
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.println("Server response: " + payload);
  } else {
    Serial.printf("HTTP Error: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }

  http.end();
  return true;
}

bool stopCamera() {
  Serial.println("Beacon away → notifying Python");
  HTTPClient http;
  http.begin(pythonServer);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST("{\"event\":\"beacon_far\"}");

  Serial.printf("HTTP Response code: %d\n", httpCode);
  if (httpCode > 0) {
    String payload = http.getString();
    Serial.println("Server response: " + payload);
  }  
  else {
    Serial.printf("HTTP Error: %s\n", http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }
   
  http.end();
  return true;
}


void loop() {
    // Provisioning portal path: no Wi-Fi creds yet, so just serve the setup page.
    if (portalActive) {
        server.handleClient();
        return;
    }

    // Checks that we still see the beacon within time out range
    if (millis() - lastBeaconTime > beaconTimeoutMs) {
        beaconFound = false;
    } else {
        beaconFound = true;
    }

    // Heartbeat: only fires while cameraActive==true (i.e. we previously
    // told the backend to start detection and it succeeded). Without this
    // gate, an ESP32 booted while the backend was down — or a beacon sitting
    // at naturally-weak RSSI — would flood the server with far events even
    // though there is nothing to shut down.
    //
    // MISSING_GRACE_MS is intentionally long because BLE/Wi-Fi coexistence
    // and pBLEScan->stop() during near can produce 10-15 s callback gaps
    // even with the beacon right next to the ESP32.
    static unsigned long lastFarHeartbeat  = 0;
    static unsigned long lastNearHeartbeat = 0;
    const unsigned long FAR_HEARTBEAT_MS  = 60000;
    const unsigned long NEAR_HEARTBEAT_MS = 60000;
    const unsigned long MISSING_GRACE_MS  = 60000;
    bool everSeen = (lastBeaconTime != 0);
    bool missing  = everSeen && (millis() - lastBeaconTime > MISSING_GRACE_MS);
    if (cameraActive && missing &&
        (lastFarHeartbeat == 0 || millis() - lastFarHeartbeat > FAR_HEARTBEAT_MS)) {
        Serial.printf("Beacon silent for %lus → heartbeat far event\n",
                      (millis() - lastBeaconTime) / 1000);
        beaconDetectedFar = true;
        if (beaconCurrentlyNear) beaconCurrentlyNear = false;
        lastFarHeartbeat = millis();
    }

    // Near heartbeat: while the beacon is still near, re-send beacon_near
    // every NEAR_HEARTBEAT_MS so the backend can recover detection state if
    // it was restarted while we thought everything was running. Backend's
    // /trigger handler is idempotent — duplicate nears are no-ops when the
    // detection subprocess is already alive.
    if (beaconCurrentlyNear &&
        (millis() - lastNearHeartbeat > NEAR_HEARTBEAT_MS)) {
        Serial.println("Beacon still near → near heartbeat");
        beaconDetectedNear = true;
        lastNearHeartbeat = millis();
    }

    //If we have data of beacon within timeout print every 1.5 seconds
    //static unsigned long lastRSSIPrint = 0;
    if (beaconFound && millis() - lastBeaconTime > 1500) {
        lastBeaconTime = millis();
        Serial.printf("KBeacon found! RSSI: %d dBm\n", lastBeaconRSSI);
    }


    // Beacon Near send HTTP Post
    if (beaconDetectedNear) {
        beaconDetectedNear = false;  // Clear flag immediately
        lastNearHeartbeat  = millis();  // suppress next heartbeat for 60 s

        Serial.println("Beacon is CLOSE (strong signal)");
        Serial.println("Beacon nearby → notifying Python");

        // NOW do the slow HTTP operation in main loop
        HTTPClient http;
        http.begin(pythonServer);
        http.addHeader("Content-Type", "application/json");
        int httpCode = http.POST("{\"event\":\"beacon_near\"}");

        Serial.printf("HTTP Response code: %d\n", httpCode);
        if (httpCode > 0) {
            String payload = http.getString();
            Serial.println("Server response: " + payload);
            cameraActive = true;
        }
        http.end();

        //Stop BLE scanning
        pBLEScan->stop();
        flagStopScan = false;
        blockUntil = millis() + 1000;
    }
    
    // If beacon far send HTTP
    if (beaconDetectedFar) {
        beaconDetectedFar = false;  // Clear flag
        
        Serial.println("Beacon is FAR (weak signal)");
        Serial.println("Beacon away → notifying Python");
        
        HTTPClient http;
        http.begin(pythonServer);
        http.addHeader("Content-Type", "application/json");
        int httpCode = http.POST("{\"event\":\"beacon_far\"}");
        
        Serial.printf("HTTP Response code: %d\n", httpCode);
        if (httpCode > 0) {
            String payload = http.getString();
            Serial.println("Server response: " + payload);
            cameraActive = false;
        }
        http.end();
    }

    // Resume scanning 1 s after a near-stop, regardless of cameraActive.
    // The previous version only resumed when cameraActive==true, which meant
    // a failed near-POST (backend down) would leave the scan stopped forever
    // — and the ESP32 would go deaf to the beacon until reboot.
    if (blockUntil != 0 && millis() > blockUntil) {
        Serial.println("Resuming BLE scan");
        pBLEScan->start(0, nullptr);
        blockUntil = 0;
    }
    
    // esp status print every 5 seconds
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 15000) {
      float seconds = millis() / 1000.0;
      Serial.printf("Time: %.1fs | Temp: %.1f°C | ", seconds, temperatureRead());
      Serial.printf("WiFi RSSI: %d dBm | ", WiFi.RSSI());
      Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
      lastPrint = millis();
    }


    //clears heap every 25 scans
    if(scans > 24){
        pBLEScan->clearResults();
        scans = 0;
    }     
}

  // BLEScanResults* foundDevices = pBLEScan->start(0, false);
  // int count = foundDevices->getCount();
  // Serial.print("Devices found: ");
  // Serial.println(count);
  // Serial.println("Scan done!");
  // pBLEScan->clearResults();   // delete results fromBLEScan buffer to release memory
  // delay(5000);

