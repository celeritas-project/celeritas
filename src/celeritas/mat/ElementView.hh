//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/ElementView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "MaterialData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access amalgamated data for an element.
 *
 * This encapsulates general data specific to an element instance. (We may
 * allow multiple element instances for an element eventually since they may
 * have various enriched abundances of component isotopes.) It also provides
 * access to ionization-related data, which in Geant4 is in a separate class.
 *
 * One use case of this class is after sampling an element in an EM model.
 *
 * The "derived quantities" may be calculated on-the-fly or stored in global
 * memory. As with the particle track view, assume that accessing them is
 * expensive and store them locally.
 */
class ElementView
{
  public:
    //!@{
    //! Type aliases
    using MaterialParamsRef = ::celeritas::NativeCRef<MaterialParamsData>;
    using AmuMass = units::AmuMass;
    //!@}

  public:
    // Construct from shared material data and global element ID
    inline CELER_FUNCTION
    ElementView(const MaterialParamsRef& params, ElementId el_id);

    //// STATIC PROPERTIES ////

    //! Atomic number Z
    CELER_FUNCTION int atomic_number() const { return def_.atomic_number; }

    //! Abundance-weighted atomic mass M [amu]
    CELER_FUNCTION AmuMass atomic_mass() const { return def_.atomic_mass; }

    //// COMPUTED PROPERTIES ////

    //! Cube root of atomic number: Z^(1/3)
    CELER_FUNCTION real_type cbrt_z() const { return def_.cbrt_z; }
    //! Cube root of Z*(Z+1)
    CELER_FUNCTION real_type cbrt_zzp() const { return def_.cbrt_zzp; }
    //! log(z)
    CELER_FUNCTION real_type log_z() const { return def_.log_z; }

    // Coulomb correction factor [unitless]
    inline CELER_FUNCTION real_type coulomb_correction() const;

    // Mass radiation coefficient for Bremsstrahlung [cm^2/g]
    inline CELER_FUNCTION real_type mass_radiation_coeff() const;

  private:
    const ElementRecord& def_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material data and global element ID.
 */
CELER_FUNCTION
ElementView::ElementView(const MaterialParamsRef& params, ElementId el_id)
    : def_(params.elements[el_id])
{
    CELER_EXPECT(el_id < params.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Coulomb correction term [cm^2/g].
 *
 * Used by Bremsstrahlung and other physics processes, this constant is
 * calculated with greater precision than in Geant4 (which is accurate to only
 * 5 or 6 digits across the range of natural elements).
 */
CELER_FUNCTION real_type ElementView::coulomb_correction() const
{
    return def_.coulomb_correction;
}

//---------------------------------------------------------------------------//
/*!
 * Mass radiation coefficient 1/X_0 for Bremsstrahlung [cm^2/g].
 *
 * The "radiation length" X_0 is "the mean distance over which a high-energy
 * electron loses all but 1/e of its energy by bremsstrahlung". See Review of
 * Particle Physics (2020), S34.4.2 (p541) for the semi-empirical calculation
 * of 1/X0.
 *
 * This quantity 1/X_0 is normalized by material density and is equivalent to
 * Geant4's \c G4Element::GetfRadTsai.
 */
CELER_FUNCTION real_type ElementView::mass_radiation_coeff() const
{
    return def_.mass_radiation_coeff;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
