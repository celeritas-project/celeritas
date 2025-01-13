//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriterInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input options for filtering ROOT MC truth output.
 *
 * This is used an input to \c celeritas::RootStepWriter::make_write_filter .
 */
struct SimpleRootFilterInput
{
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    std::vector<size_type> track_id;
    size_type event_id = unspecified;
    size_type parent_id = unspecified;
    size_type action_id = unspecified;

    //! True if any filtering is being applied
    explicit operator bool() const
    {
        return !track_id.empty() || event_id != unspecified
               || parent_id != unspecified || action_id != unspecified;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
