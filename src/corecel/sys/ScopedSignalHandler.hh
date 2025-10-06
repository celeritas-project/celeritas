//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedSignalHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <initializer_list>
#include <utility>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Catch the given signal type within the scope of the handler.
 *
 * On instantiation with a non-empty argument, this class registers a signal
 * handler for the given signal. A class instance is true if and only if the
 * class is handling a signal. The instance's "call" operator will check and
 * return whether the assigned signal has been caught. The move-assign operator
 * can be used to unregister the handle.
 *
 * When the class exits scope, the signal for the active type will be cleared.
 *
 * Signal handling can be disabled by setting the environment variable \c
 * CELER_DISABLE_SIGNALS flag, but hopefully this will not be necessary because
 * signal handling should be used sparingly.
 *
 * \code
   #include <csignal>

   int main()
   {
      ScopedSignalHandler interrupted(SIGINT);

      while (true)
      {
          if (interrupted())
          {
              CELER_LOG(error) << "Interrupted";
              break;
          }

          if (stop_handling_for_whatever_reason())
          {
              // Clear handler
              interrupted = {};
          }
      }
      return interrupted() ? 1 : 0;
   }
   \endcode
 *
 * \warning This class is not thread safe. If multiple threads have this in
 * scope, only one active (and indeterminate!) thread will mark the flag as
 * intercepted.
 */
class ScopedSignalHandler
{
  public:
    //!@{
    //! \name Type aliases
    using signal_type = int;
    //!@}

  public:
    // Whether signals are enabled
    static bool enabled();

    // Raise a signal visible only to ScopedSignalHandler (for testing)
    static int raise(signal_type sig);

    //! Default to not handling any signals.
    ScopedSignalHandler() = default;

    // Handle the given signal type, asserting if it's already been raised
    explicit ScopedSignalHandler(signal_type);

    // Handle the given signal types
    explicit ScopedSignalHandler(std::initializer_list<signal_type>);

    // Release the given signal
    ~ScopedSignalHandler();

    // Check if signal was intercepted
    inline bool operator()() const;

    //! True if handling a signal
    explicit operator bool() const { return mask_ != 0; }

    // Move construct and assign to capture/release signal handling
    ScopedSignalHandler(ScopedSignalHandler const&) = delete;
    ScopedSignalHandler& operator=(ScopedSignalHandler const&) = delete;
    ScopedSignalHandler(ScopedSignalHandler&&) noexcept;
    ScopedSignalHandler& operator=(ScopedSignalHandler&&) noexcept;
    void swap(ScopedSignalHandler& other) noexcept;

  private:
    using HandlerPtr = void (*)(int);
    using PairSigHandle = std::pair<signal_type, HandlerPtr>;
    using VecSH = std::vector<PairSigHandle>;

    signal_type mask_{0};
    VecSH handles_;

    bool check_signal() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return whether a signal was intercepted.
 */
bool ScopedSignalHandler::operator()() const
{
    return *this && this->check_signal();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
