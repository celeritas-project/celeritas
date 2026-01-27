//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/ScoringParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/Scoring.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    ScoringParams ...;
   \endcode
 */
class ScoringParams
{
  public:
    //!@{
    //! \name Type aliases
    using HitCallbackFunc = inp::OpticalScoring::HitCallbackFunc;
    //!@}

  public:
    ScoringParams(inp::OpticalScoring);

  private:
    std::optional<HitCallbackFunc> detector_callback_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
