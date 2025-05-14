/*
Control and Power Board Startup Program
4/2/25
Laura Helmich

This program toggles Teensy pins in order to test if 
each part of the power and control board is working 
properly. More explanation in "Testing Procedure" 
google doc (Spring 25 -> Electrical folder)
*/

//GPIO pin definitions:
#define _12V_ON_H 4
#define _5V_CAM_ON_H 7
#define _3V3_ON_H 5
#define _3V3_PG_H 6
#define V_ADJ_ON_H 8
#define UP_FLASH_H 11
#define Cam_long_H 34
#define Cam_short_H 33
#define Pet_dog_H 20

void setup() {
  //set up Teensy LED
  pinMode(LED_BUILTIN, OUTPUT);

  //set up Serial communication
  Serial.begin(9600);               // Start Serial communication

  pinMode(_12V_ON_H, OUTPUT);       // Set each pin as an output
  pinMode(_5V_CAM_ON_H, OUTPUT); 
  pinMode(_3V3_ON_H, OUTPUT); 
  pinMode(_3V3_PG_H, OUTPUT); 
  pinMode(V_ADJ_ON_H, OUTPUT); 
  pinMode(UP_FLASH_H, OUTPUT);
  pinMode(Cam_long_H, OUTPUT);
  pinMode(Cam_short_H, OUTPUT); 
  pinMode(Pet_dog_H, OUTPUT);

  digitalWrite(_12V_ON_H, LOW);     // Ensure all pins are OFF initially
  digitalWrite(_5V_CAM_ON_H, LOW); 
  digitalWrite(_3V3_ON_H, LOW); 
  digitalWrite(_3V3_PG_H, LOW); 
  digitalWrite(V_ADJ_ON_H, LOW); 
  digitalWrite(UP_FLASH_H, LOW);
  digitalWrite(Cam_long_H, LOW);
  digitalWrite(Cam_short_H, LOW); 
  digitalWrite(Pet_dog_H, LOW);

  Serial.println("Ready. Type '(signal name) on' or '(signal name) off' to toggle pins.\n");
  Serial.println("Signal names: \n");
  Serial.println("12V_ON_H      ->      12v \n");
  Serial.println("5V_CAM_ON_H   ->      5v \n");
  Serial.println("_3V3_ON_H     ->      3.3v \n");
  Serial.println("_3V3_PG_H     ->      3.3v pg \n");
  Serial.println("V_ADJ_ON_H    ->      HV \n");
  Serial.println("UP_FLASH_H    ->      flash \n");
  Serial.println("Cam_long_H    ->      cam long \n");
  Serial.println("Cam_short_H   ->      cam short \n");
  Serial.println("Pet_dog_H     ->      pet dog \n");
  Serial.println("\n");
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH); //Teensy LED turns on if it has power

  if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');  // Read input
        command.trim();  // Remove any trailing newline or whitespace

        //on commands:
        if (command.equalsIgnoreCase("12V on")) {
            digitalWrite(_12V_ON_H, HIGH);
            Serial.println("12V_ON_H turned ON");
        }
        else if (command.equalsIgnoreCase("5V on")) {
            digitalWrite(_5V_CAM_ON_H, HIGH);
            Serial.println("5V_CAM_ON_H turned ON");
        }
        else if (command.equalsIgnoreCase("3.3v on")) {
            digitalWrite(_3V3_ON_H, HIGH);
            Serial.println("3V3_ON_H turned ON");
        }
        else if (command.equalsIgnoreCase("3.3v pg on")) {
            digitalWrite(_3V3_PG_H, HIGH);
            Serial.println("3V3_PG_H turned ON");
        }
        else if (command.equalsIgnoreCase("HV on")) {
            digitalWrite(V_ADJ_ON_H, HIGH);
            Serial.println("V_ADJ_ON_H turned ON");
        }
        else if (command.equalsIgnoreCase("flash on")) {
            digitalWrite(UP_FLASH_H, HIGH);
            Serial.println("UP_FLASH_H turned ON");
        }
        else if (command.equalsIgnoreCase("cam long on")) {
            digitalWrite(Cam_long_H, HIGH);
            Serial.println("Cam_long_H turned ON");
        }
        else if (command.equalsIgnoreCase("cam short on")) {
            digitalWrite(Cam_short_H, HIGH);
            Serial.println("Cam_short_H turned ON");
        }
        else if (command.equalsIgnoreCase("pet dog on")) {
            digitalWrite(Pet_dog_H, HIGH);
            Serial.println("Pet_dog_H turned ON");
        }


        //off commands
        else if (command.equalsIgnoreCase("12V off")) {
            digitalWrite(_12V_ON_H, LOW);
            Serial.println("_12V_ON_H turned OFF");
        }
        else if (command.equalsIgnoreCase("5V off")) {
            digitalWrite(_5V_CAM_ON_H, LOW);
            Serial.println("5V_CAM_ON_H turned OFF");
        }
        else if (command.equalsIgnoreCase("3.3v off")) {
            digitalWrite(_3V3_ON_H, LOW);
            Serial.println("3V3_ON_H turned OFF");
        }
        else if (command.equalsIgnoreCase("3.3v pg off")) {
            digitalWrite(_3V3_PG_H, LOW);
            Serial.println("3V3_PG_H turned OFF");
        }
        else if (command.equalsIgnoreCase("HV off")) {
            digitalWrite(V_ADJ_ON_H, LOW);
            Serial.println("V_ADJ_ON_H turned OFF");
        }
        else if (command.equalsIgnoreCase("flash off")) {
            digitalWrite(UP_FLASH_H, LOW);
            Serial.println("UP_FLASH_H turned OFF");
        }
        else if (command.equalsIgnoreCase("cam long off")) {
            digitalWrite(Cam_long_H, LOW);
            Serial.println("Cam_long_H turned OFF");
        }
        else if (command.equalsIgnoreCase("cam short off")) {
            digitalWrite(Cam_short_H, LOW);
            Serial.println("Cam_short_H turned OFF");
        }
        else if (command.equalsIgnoreCase("pet dog off")) {
            digitalWrite(Pet_dog_H, LOW);
            Serial.println("Pet_dog_H turned OFF");
        }
        else {
            Serial.println("Unknown command. Type '(signal name) on' or '(signal name) off'.");
        }
    }

}
