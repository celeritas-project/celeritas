//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/NonMasterResultPrinter.cc
//---------------------------------------------------------------------------//
#include "NonMasterResultPrinter.hh"

#include <iostream>
#include <sstream>

#include "corecel/io/ColorUtils.hh"

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with MPI rank.
 */
NonMasterResultPrinter::NonMasterResultPrinter(int rank) : rank_(rank) {}

//---------------------------------------------------------------------------//
/*!
 * Print output.
 */
void NonMasterResultPrinter::OnTestPartResult(
    ::testing::TestPartResult const& result)
{
    // If the test part succeeded, we don't need to do anything.
    if (result.type() == ::testing::TestPartResult::kSuccess)
        return;

    // Build an independent string stream so that the whole content is more
    // likely to be flushed at once
    std::ostringstream os;
    os << color_code('r') << "[  FAILED  ]" << color_code(' ');

    if (result.file_name())
    {
        os << result.file_name() << ":";
    }
    if (result.line_number() >= 0)
    {
        os << result.line_number() << ":";
    }
    os << " Failure on rank " << rank_ << ":\n" << result.message();

    std::cout << os.str() << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
