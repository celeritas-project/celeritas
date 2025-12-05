//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/EmExtraPhysicsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

class G4GammaNuclearXS;

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * A helper class to interface with Geant4 cross sections.
 */
class EmExtraPhysicsHelper
{
  public:
    // Construct EM extra physics helper
    EmExtraPhysicsHelper();

    // Calculate gamma-nuclear element cross section
    double GammaNuclearElementXS(double energy, int z);

    // The maximum high energy of G4PhotoNuclearCrossSection
    static constexpr double max_high_energy()
    {
        return 5e+4;  // clhep::MeV
    }

  private:
    //// DATA ////
    std::shared_ptr<G4GammaNuclearXS> gn_xs_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
