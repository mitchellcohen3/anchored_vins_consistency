#ifndef OV_CORE_TIMER_H
#define OV_CORE_TIMER_H

#include <chrono>

namespace ov_core {

/**
 * @brief Simple start/stop timer using std::chrono::steady_clock.
 *
 * Usage:
 * @code
 *   Timer timer;
 *   timer.start();
 *   // ... code to time ...
 *   timer.stop();
 *   double dt = timer.elapsed(); // seconds
 * @endcode
 */
class Timer {
public:
  using clock = std::chrono::steady_clock;

  Timer() = default;

  void start() { _start = clock::now(); }

  void stop() { _end = clock::now(); }

  /// Returns elapsed time in seconds between the last start() and stop() calls.
  double elapsed() const { return std::chrono::duration<double>(_end - _start).count(); }

private:
  clock::time_point _start;
  clock::time_point _end;
};

} // namespace ov_core

#endif // OV_CORE_TIMER_H
