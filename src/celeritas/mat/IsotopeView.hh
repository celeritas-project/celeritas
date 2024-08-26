//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/IsotopeView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "MaterialData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access amalgamated data for a nuclide (isotope of an element).
 *
 * This encapsulates general data specific to a single nuclide.
 *
 * \todo This will be renamed to NuclideView.
 */
class IsotopeView
{
  public:
    //!@{
    //! \name Type aliases
    using MaterialParamsRef = NativeCRef<MaterialParamsData>;
    using MevEnergy = units::MevEnergy;
    using MevMass = units::MevMass;
    using AtomicMassNumber = AtomicNumber;
    //!@}

  public:
    // Construct from shared material data and global isotope ID
    inline CELER_FUNCTION
    IsotopeView(MaterialParamsRef const& params, IsotopeId isot_id);

    // ID of this isotope
    CELER_FORCEINLINE_FUNCTION IsotopeId isotope_id() const;

    // Atomic number Z
    CELER_FORCEINLINE_FUNCTION AtomicNumber atomic_number() const;

    // Atomic number A
    CELER_FORCEINLINE_FUNCTION AtomicMassNumber atomic_mass_number() const;

    // Nuclear binding energy
    CELER_FORCEINLINE_FUNCTION MevEnergy binding_energy() const;

    // Nuclear binding energy difference for a proton loss
    CELER_FORCEINLINE_FUNCTION MevEnergy proton_loss_energy() const;

    // Nuclear binding energy difference for a neutron loss
    CELER_FORCEINLINE_FUNCTION MevEnergy neutron_loss_energy() const;

    // Sum of nucleons + binding energy
    CELER_FORCEINLINE_FUNCTION MevMass nuclear_mass() const;

  private:
    MaterialParamsRef const& params_;
    IsotopeId isotope_;

    // HELPER FUNCTIONS

    CELER_FORCEINLINE_FUNCTION IsotopeRecord const& isotope_def() const
    {
        return params_.isotopes[isotope_];
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material data and global isotope ID.
 */
CELER_FUNCTION
IsotopeView::IsotopeView(MaterialParamsRef const& params, IsotopeId isot_id)
    : params_{params}, isotope_{isot_id}
{
    CELER_EXPECT(isotope_ < params.isotopes.size());
}

//---------------------------------------------------------------------------//
/*!
 * Isotope ID.
 */
CELER_FUNCTION IsotopeId IsotopeView::isotope_id() const
{
    return isotope_;
}

//---------------------------------------------------------------------------//
/*!
 * Atomic number Z.
 */
CELER_FUNCTION AtomicNumber IsotopeView::atomic_number() const
{
    return isotope_def().atomic_number;
}

//---------------------------------------------------------------------------//
/*!
 * Atomic number A.
 */
CELER_FUNCTION IsotopeView::AtomicMassNumber
IsotopeView::atomic_mass_number() const
{
    return isotope_def().atomic_mass_number;
}

//---------------------------------------------------------------------------//
/*!
 * Nuclear binding energy.
 */
CELER_FUNCTION units::MevEnergy IsotopeView::binding_energy() const
{
    return isotope_def().binding_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Nuclear binding energy difference for a proton loss.
 */
CELER_FUNCTION units::MevEnergy IsotopeView::proton_loss_energy() const
{
    return isotope_def().proton_loss_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Nuclear binding energy difference for a neutron loss.
 */
CELER_FUNCTION units::MevEnergy IsotopeView::neutron_loss_energy() const
{
    return isotope_def().neutron_loss_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Nuclear mass, the sum of the nucleons' mass and their binding energy.
 */
CELER_FUNCTION units::MevMass IsotopeView::nuclear_mass() const
{
    return isotope_def().nuclear_mass;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
