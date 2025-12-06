//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/EmExtraPhysicsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

class G4GammaNuclearXS;

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * A helper class for interfacing with Geant4 cross section calculations and
 * other properties.
 *
 * This class primarily severs as a wrapper around Geant4 cross section
 * calculation methods, which are not directly accessible from Celeritas EM
 * physics models. Use of this class requires CELERITAS_USE_GEANT4 to be
 * enabled.
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

    // Calculate gamma-nuclear element cross section
    MmSqXs calc_gamma_nuclear_xs(AtomicNumber z, MevEnergy energy) const;

  private:
    //// DATA ////
    std::shared_ptr<G4GammaNuclearXS> gn_xs_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELERITAS_USE_GEANT4
inline EmExtraPhysicsHelper::EmExtraPhysicsHelper()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline auto
EmExtraPhysicsHelper::calc_gamma_nuclear_xs(AtomicNumber z,
                                            MevEnergy energy)() const -> MmSqXs
{
    CELER_ASSERT_UNREACHABLE();
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
