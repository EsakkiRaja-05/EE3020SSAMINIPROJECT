#include <esp_camera.h>
#include <WiFi.h>
#include <TensorFlowLite_ESP32.h>   // YOLO / Edge Impulse model
#include "model_yolo.h"
#include <Wire.h>
#include <Adafruit_SSD1306.h>

// OLED setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Camera configuration
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// WiFi credentials
const char* ssid = "YourWiFiSSID";
const char* password = "YourWiFiPassword";

// YOLO model variables
TfLiteModel* model;
TfLiteInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);

  // WiFi init
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected!");

  // Camera init
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed!");
    return;
  }

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
    for (;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Load YOLO model
  model = tflite_load_model(model_yolo);
  interpreter = tflite_create_interpreter(model);
  tflite_allocate_tensors(interpreter);
  input = tflite_input_tensor(interpreter, 0);
  output = tflite_output_tensor(interpreter, 0);

  Serial.println("YOLO model loaded successfully!");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // Preprocess image for YOLO
  preprocess_image(fb->buf, fb->len, input);

  // Run inference
  tflite_invoke(interpreter);

  // Get prediction
  int detected_class = get_detected_class(output);
  String label = "";
  if (detected_class == 0) label = "PET Bottle";
  else if (detected_class == 1) label = "HDPE Container";
  else if (detected_class == 2) label = "LDPE Bag";
  else label = "Unknown";

  // Display result
  display.clearDisplay();
  display.setCursor(0, 10);
  display.print("Detected: ");
  display.println(label);
  display.display();

  Serial.println("Detected Plastic Type: " + label);

  esp