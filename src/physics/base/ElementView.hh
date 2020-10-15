//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "MaterialParamsPointers.hh"
#include "Units.hh"

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
    //@{
    //! Type aliases
    using AmuMass = units::AmuMass;
    //@}

  public:
    // Construct with defaults
    ElementView(const MaterialParamsPointers& params, ElementDefId el_id);

    // >>> STATIC PROPERTIES

    //! Atomic number Z
    CELER_FUNCTION int atomic_number() const { return el_.atomic_number; }

    //! Abundance-weighted atomic mass M [amu]
    CELER_FUNCTION AmuMass atomic_mass() const { return el_.atomic_mass; }

    // >>> DERIVED PROPERTIES

    //! Radiation length for Bremsstrahlung [cm]
    CELER_FUNCTION real_type radiation_length_tsai() const
    {
        return el_.radiation_length_tsai;
    }

    //! Cube root of atomic number: Z^(1/3)
    CELER_FUNCTION real_type cbrt_z() const { return el_.cbrt_z; }
    //! Cube root of Z*(Z+1)
    CELER_FUNCTION real_type cbrt_zzp() const { return el_.cbrt_zzp; }
    //! log(z)
    CELER_FUNCTION real_type log_z() const { return el_.log_z; }

  private:
    const ElementDef& el_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
