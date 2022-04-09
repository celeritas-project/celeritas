//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedSignalHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedSignalHandler.hh"

#include <csignal>

#include "base/Assert.hh"

#include "Environment.hh"
#include "Logger.hh"

namespace
{
//---------------------------------------------------------------------------//
// Use an environment variable to disable signals
bool determine_allow_signals()
{
    if (!celeritas::getenv("CELER_DISABLE_SIGNALS").empty())
    {
        CELER_LOG(info)
            << "Disabling signal support since the 'CELER_DISABLE_SIGNALS' "
               "environment variable is present and non-empty";
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
// Bitset of signals that have been called
volatile sig_atomic_t celer_signal_bits = 0;

//---------------------------------------------------------------------------//
//! Set the bit corresponding to a signal
extern "C" void celer_set_signal(int signal)
{
    CELER_ASSERT(signal >= 0 && signal < static_cast<int>(sizeof(int) * 8 - 1));
    celer_signal_bits |= (1 << signal);
}

//---------------------------------------------------------------------------//
//! Clear the bit corresponding to a signal
void celer_clr_signal(int signal)
{
    celer_signal_bits &= ~(1 << signal);
}

//---------------------------------------------------------------------------//
//! Set the bit corresponding to a signal
bool celer_chk_signal(int signal)
{
    return celer_signal_bits & (1 << signal);
}

//---------------------------------------------------------------------------//
} // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether signal handling is enabled.
 */
bool ScopedSignalHandler::allow_signals()
{
    static bool result = determine_allow_signals();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Handle the given signal type.
 */
ScopedSignalHandler::ScopedSignalHandler(signal_type sig)
{
    CELER_EXPECT(sig >= 0);

    if (!ScopedSignalHandler::allow_signals())
    {
        // Signal handling is disabled and an info message has already been
        // displayed
        return;
    }

    CELER_VALIDATE(!celer_chk_signal(sig),
                   << "unhandled signal " << sig
                   << " when creating new signal handler");

    // Register signal
    signal_      = sig;
    prev_handle_ = std::signal(signal_, celer_set_signal);

    CELER_ENSURE(prev_handle_ != SIG_ERR);
}

//---------------------------------------------------------------------------//
/*!
 * Release the given signal.
 */
ScopedSignalHandler::~ScopedSignalHandler()
{
    if (signal_ >= 0)
    {
        // Clear signal bit
        celer_clr_signal(signal_);
        // Restore signal handler
        std::signal(signal_, prev_handle_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Move construct.
 */
ScopedSignalHandler::ScopedSignalHandler(ScopedSignalHandler&& other) noexcept
{
    this->swap(other);
}

//---------------------------------------------------------------------------//
/*!
 * Move assign.
 */
ScopedSignalHandler&
ScopedSignalHandler::operator=(ScopedSignalHandler&& other) noexcept
{
    ScopedSignalHandler temp(std::move(other));
    this->swap(temp);
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Swap.
 */
void ScopedSignalHandler::swap(ScopedSignalHandler& other) noexcept
{
    using std::swap;
    swap(signal_, other.signal_);
    swap(prev_handle_, other.prev_handle_);
}

//---------------------------------------------------------------------------//
/*!
 * True if signal was intercepted.
 */
bool ScopedSignalHandler::check_signal() const
{
    return celer_chk_signal(signal_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
