//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedTimeAndRedirect.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "base/ScopedStreamRedirect.hh"
#include "base/ScopedTimeLog.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * At end of scope, print elapsed time and captured cout/cerr.
 *
 * This is designed to prevent other libraries (Geant4,VecGeom) from polluting
 * stdout and breaking JSON reading ability.
 *
 * \code
    {
        ScopedTimeAndRedirect temp_;
        vecgeom::DoNoisyAndLongStuff();
    }
   \endcode
 *
 * \warning During scope, you should be sure *NOT* to call the logger, which by
 * default prints to cerr.
 */
class ScopedTimeAndRedirect
{
  public:
    explicit ScopedTimeAndRedirect(std::string label);
    ~ScopedTimeAndRedirect();

  private:
    std::unique_ptr<ScopedStreamRedirect> stdout_;
    std::unique_ptr<ScopedStreamRedirect> stderr_;
    std::string label_;
    ScopedTimeLog                         scoped_time_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
