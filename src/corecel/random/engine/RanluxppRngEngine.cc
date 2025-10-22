//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/RanluxppRngEngine.cc
//---------------------------------------------------------------------------//
#include "RanluxppRngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Skip `n` random numbers without generating them
 */
CELER_FUNCTION void RanluxppRngEngine::skip(RanluxppUInt n)
{
    CELER_ASSERT(n > 0);
    CELER_ASSERT(params_.kMaxPos > 0);

    int left = (params_.kMaxPos - state_->position) / offset_;
    CELER_ASSERT(left >= 0);
    if (n < static_cast<RanluxppUInt>(left))
    {
        // Just skip the next few entries in the currently
        // available bits.
        state_->position += n * offset_;
        CELER_ASSERT(state_->position <= params_.kMaxPos);
        return;
    }

    n -= left;
    // Need to advance and possibly skip over blocks.
    int nPerState = params_.kMaxPos / offset_;
    int skip = n / nPerState;

    RanluxppArray9 a_skip
        = celeritas::detail::compute_power_modulus(params_.kA_2048, skip + 1);

    RanluxppArray9 lcg
        = celeritas::detail::to_lcg(state_->state, state_->carry);
    lcg = celeritas::detail::compute_mod_multiply(a_skip, lcg);
    state_->state = celeritas::detail::to_ranlux(lcg, state_->carry);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * nPerState;
    CELER_ASSERT(remaining >= 0);
    state_->position = remaining * offset_;
    CELER_ASSERT(state_->position <= params_.kMaxPos);
}

//---------------------------------------------------------------------------//
/*!
 * Return the next random bits, generate a new block if necessary
 */
CELER_FUNCTION RanluxppUInt RanluxppRngEngine::nextRandomBits()
{
    if (state_->position + offset_ > params_.kMaxPos)
    {
        this->advance();
    }

    int idx = state_->position / 64;
    int offset = state_->position % 64;
    int numBits = 64 - offset;

    RanluxppUInt bits = state_->state[idx] >> offset;
    if (numBits < offset_)
    {
        bits |= state_->state[idx + 1] << numBits;
    }
    bits &= ((RanluxppUInt(1) << offset_) - 1);

    state_->position += offset_;
    CELER_ASSERT(state_->position <= params_.kMaxPos);

    return bits;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
