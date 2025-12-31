//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/EmExtraPhysicsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/math/Quantity.hh"
#include "corecel/math/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/phys/AtomicNumber.hh"

class G4DynamicParticle;
class G4ElectroNuclearCrossSection;
class G4GammaNuclearXS;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate Geant4 gamma-nuclear and electro-nuclear cross sections.
 *
 * This class primarily severs as a wrapper around Geant4 cross section
 * calculation methods, which are not directly accessible from Celeritas EM
 * physics models. Use of this class requires Geant4 11.0 or higher.
 */
class EmExtraPhysicsHelper
{
  public:
    //!@{
    using MevEnergy = units::MevEnergy;
    using MmSqXs
        = Quantity<UnitProduct<units::Millimeter, units::Millimeter>, double>;
    //!@}

  public:
    // Construct EM extra physics helper
    EmExtraPhysicsHelper();

    // Calculate electro-nuclear element cross section
    MmSqXs calc_electro_nuclear_xs(AtomicNumber z, MevEnergy energy) const;

    // Calculate gamma-nuclear element cross section
    MmSqXs calc_gamma_nuclear_xs(AtomicNumber z, MevEnergy energy) const;

  private:
    //// DATA ////
    std::shared_ptr<G4DynamicParticle> particle_;
    std::shared_ptr<G4ElectroNuclearCrossSection> en_xs_;
    std::shared_ptr<G4GammaNuclearXS> gn_xs_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if CELERITAS_GEANT4_VERSION < 0x0b0000
inline EmExtraPhysicsHelper::EmExtraPhysicsHelper()
{
#    if !CELERITAS_USE_GEANT4
    CELER_NOT_CONFIGURED("Geant4");
#    else
    CELER_VALIDATE(false,
                   << "Geant4 version " << cmake::geant4_version
                   << " is too old for gamma-nuclear cross section "
                      "calculation");
#    endif
}

inline EmExtraPhysicsHelper::MmSqXs
EmExtraPhysicsHelper::calc_electro_nuclear_xs(AtomicNumber, MevEnergy) const
{
    CELER_ASSERT_UNREACHABLE();
}

inline EmExtraPhysicsHelper::MmSqXs
EmExtraPhysicsHelper::calc_gamma_nuclear_xs(AtomicNumber, MevEnergy) const
{
    CELER_ASSERT_UNREACHABLE();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
