//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/ScoringParams.cc
//---------------------------------------------------------------------------//
#include "ScoringParams.hh"

#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"

#include "DetectorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
ScoringParams::ScoringParams(inp::OpticalScoring input)
    : detector_callback_(std::move(input.detector_callback))
{
    if (detector_callback_)
    {
        CELER_LOG(info) << "optical scoring enabled.";
    }
    else
    {
        CELER_LOG(info) << "optical scoring disabled.";
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
