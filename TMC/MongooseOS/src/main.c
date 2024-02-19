#include <stdio.h>
#include <string.h>
#include "mgos.h"
#include "mgos_mqtt.h"
int i = 0;
static void my_timer_cb(void *arg) {
  char msg[80];
  sprintf(msg, "Good job! You are connected %d times", i);
  i++;
  mgos_mqtt_pub("/esp8266", msg, strlen(msg), 1, 0);
  (void) arg;
}
enum mgos_app_init_result mgos_app_init(void) {
  mgos_set_timer(5000, MGOS_TIMER_REPEAT, my_timer_cb, NULL);
  return MGOS_APP_INIT_SUCCESS;
}

