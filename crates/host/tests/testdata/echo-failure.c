/* Test-only ABI-3 failure fixture. No DSP or acoustic-quality claim.
 * The mounted test launches a separate host with this explicit library path.
 * Build once: cc -dynamiclib -o /tmp/speakerdesk-echo-failure.dylib echo-failure.c
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

struct state { unsigned calls; int process_failure; int reset_failure; };
static int mode(const char *expected) {
    const char *value = getenv("SPEAKERDESK_TEST_ECHO_FAILURE");
    return value && strcmp(value, expected) == 0;
}
int speakerdesk_echo_abi(void) { return mode("abi") ? 2 : 3; }
void *speakerdesk_echo_create(void) {
    struct state *state = calloc(1, sizeof(*state));
    if (state) {
        state->process_failure = mode("process");
        state->reset_failure = mode("reset");
    }
    return state;
}
void speakerdesk_echo_destroy(void *value) { free(value); }
int speakerdesk_echo_reset(void *value) {
    struct state *state = value;
    state->calls = 0;
    return state->reset_failure ? -1 : 0;
}
int speakerdesk_echo_process(void *value, const float *render,
                            const float *microphone, float *output,
                            size_t frames, int delay) {
    (void)render; (void)delay;
    struct state *state = value;
    if (frames != 160 || (state->process_failure && ++state->calls >= 100)) return -1;
    memcpy(output, microphone, frames * sizeof(float));
    return 0;
}
