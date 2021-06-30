int switchpin=8;
int switchvalue;
int optopin=1;
int ledpin=5;
long max_iteration;

int sync_LED_pin = 6;
int sync_LED_pin2 = 7;
int sync_OE_pin = 3;

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(switchpin,INPUT_PULLUP);
  pinMode(optopin, OUTPUT);
  pinMode(ledpin, OUTPUT);
  
  pinMode(sync_LED_pin, OUTPUT);
  pinMode(sync_LED_pin2, OUTPUT);
  pinMode(sync_OE_pin, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
  switchvalue=digitalRead(switchpin);
  if(switchvalue==LOW){
    opto_stimulation();
  }else if(switchvalue==HIGH){
    digitalWrite(sync_LED_pin, HIGH);
    digitalWrite(sync_LED_pin2, HIGH);

    digitalWrite(sync_OE_pin, HIGH);
    delay(150);
    digitalWrite(sync_LED_pin, LOW);
    digitalWrite(sync_LED_pin2, LOW);
    digitalWrite(sync_OE_pin, LOW);

    max_iteration = random(20, 60);
    for(int sync=0; sync<max_iteration;sync++){
      switchvalue=digitalRead(switchpin);
      if(switchvalue==LOW){
        opto_stimulation();
      }else if(switchvalue==HIGH){
        delay(1000);    
      }
      
    }    
    switchvalue=digitalRead(switchpin);
    }   
}

void opto_stimulation() {
  opto_identification();
  delay(15000);
  opto_identification();
  fifty_Hz_stimulation(100);
  delay(5000);
  fifty_Hz_stimulation(200);
  }


 void opto_identification(){
       for(int x=0; x<100;x++){
        digitalWrite(LED_BUILTIN, HIGH);
        digitalWrite(ledpin,HIGH);
        digitalWrite(optopin, HIGH);
        delay(3); 
        digitalWrite(optopin, LOW);
        delay(27);
        digitalWrite(LED_BUILTIN, LOW);
        digitalWrite(ledpin, LOW);
        delay(1970); 
        }    
  } 

 void fifty_Hz_stimulation(int duration){
        int number_of_iterations;
        number_of_iterations = 5*duration/100;
        for(int y=0;y<5;y++){
        for(int x=0; x<number_of_iterations;x++){
          digitalWrite(LED_BUILTIN, HIGH);
          digitalWrite(ledpin,HIGH);
          digitalWrite(optopin, HIGH);
          delay(3); 
          digitalWrite(optopin, LOW);
          digitalWrite(LED_BUILTIN, LOW);
          digitalWrite(ledpin, LOW);
          delay(17); 
          }
          delay(duration);} 
  
  }
