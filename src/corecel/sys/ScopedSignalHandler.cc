//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedSignalHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedSignalHandler.hh"

#include <algorithm>
#include <csignal>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

namespace
{
//---------------------------------------------------------------------------//
// Bitset of signals that have been called
sig_atomic_t volatile g_celer_signal_bits_ = 0;

//---------------------------------------------------------------------------//
//! Set the bit corresponding to a signal
extern "C" void celer_set_signal(int signal)
{
    CELER_ASSERT(signal >= 0 && signal < static_cast<int>(sizeof(int) * 8 - 1));
    g_celer_signal_bits_ |= (1 << signal);
}

//---------------------------------------------------------------------------//
//! Clear the bit(s) corresponding to one or more signals
void celer_clr_signal(int mask)
{
    g_celer_signal_bits_ &= ~mask;
}

//---------------------------------------------------------------------------//
//! Return whether the bit corresponding to any of the given signalsis set
bool celer_chk_signal(int mask)
{
    return g_celer_signal_bits_ & mask;
}

//---------------------------------------------------------------------------//
}  // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether signal handling is enabled.
 */
bool ScopedSignalHandler::allow_signals()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_DISABLE_SIGNALS").empty())
        {
            CELER_LOG(info) << "Disabling signal support since the "
                               "'CELER_DISABLE_SIGNALS' "
                               "environment variable is present and non-empty";
            return false;
        }
        return true;
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Raise a signal visible only to ScopedSignalHandler (for testing).
 *
 * This function always returns zero, indicating "success", to conform to the
 * C++ standard library function.
 */
int ScopedSignalHandler::raise(signal_type sig)
{
    celer_set_signal(sig);
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Handle the given signal type.
 */
ScopedSignalHandler::ScopedSignalHandler(signal_type sig)
    : ScopedSignalHandler({sig})
{
    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Handle the given signal type.
 */
ScopedSignalHandler::ScopedSignalHandler(
    std::initializer_list<signal_type> signals)
{
    CELER_EXPECT(signals.begin() != signals.end());
    CELER_EXPECT(std::all_of(signals.begin(),
                             signals.end(),
                             [](signal_type sig) { return sig >= 0; }));

    if (!ScopedSignalHandler::allow_signals())
    {
        // Signal handling is disabled and an info message has already been
        // displayed
        return;
    }

    for (signal_type sig : signals)
    {
        mask_ |= (1 << sig);

        CELER_VALIDATE(!celer_chk_signal(mask_),
                       << "unhandled signal " << sig
                       << "existed when creating new signal handler");

        // Register signal
        HandlerPtr prev_handle = std::signal(sig, celer_set_signal);
        CELER_ASSERT(prev_handle != SIG_ERR);
        handles_.push_back({sig, prev_handle});
    }

    CELER_ENSURE(handles_.size() == signals.size());
    CELER_ENSURE(*this);
}

//---------------------------------------------------------------------------//
/*!
 * Release the given signal.
 *
 * This destructor is not thread-safe; it could have a race condition if
 * a signal is sent while our signal bit is being cleared.
 */
ScopedSignalHandler::~ScopedSignalHandler()
{
    for (auto const& sig_handle : handles_)
    {
        // Restore signal handler
        std::signal(sig_handle.first, sig_handle.second);
    }
    // Clear signal bits
    celer_clr_signal(mask_);
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
    swap(mask_, other.mask_);
    swap(handles_, other.handles_);
}

//---------------------------------------------------------------------------//
/*!
 * True if signal was intercepted.
 */
bool ScopedSignalHandler::check_signal() const
{
    return celer_chk_signal(mask_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
