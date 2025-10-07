//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/ElementView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "IsotopeView.hh"
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
 *
 * \note For testing purposes, isotopic data may be unavailable.
 */
class ElementView
{
  public:
    //!@{
    //! \name Type aliases
    using MaterialParamsRef = NativeCRef<MaterialParamsData>;
    using AmuMass = units::AmuMass;
    //!@}

  public:
    // Construct from shared material data and global element ID
    inline CELER_FUNCTION
    ElementView(MaterialParamsRef const& params, ElementId el_id);

    //// STATIC PROPERTIES ////

    // Atomic number Z
    CELER_FORCEINLINE_FUNCTION AtomicNumber atomic_number() const;

    //! Abundance-weighted atomic mass M [amu]
    CELER_FUNCTION AmuMass atomic_mass() const { return def_.atomic_mass; }

    // Number of isotopic components
    inline CELER_FUNCTION IsotopeComponentId::size_type num_isotopes() const;

    // View properties of a specific isotope
    inline CELER_FUNCTION IsotopeView isotope_record(IsotopeComponentId id) const;

    // ID of an isotope component of this element
    inline CELER_FUNCTION IsotopeId isotope_id(IsotopeComponentId id) const;

    // Advanced access to the element's isotopes (id/fraction)
    inline CELER_FUNCTION Span<ElIsotopeComponent const> isotopes() const;

    //// COMPUTED PROPERTIES ////

    //! Cube root of atomic number: Z^(1/3)
    CELER_FUNCTION real_type cbrt_z() const { return def_.cbrt_z; }
    //! Cube root of Z*(Z+1)
    CELER_FUNCTION real_type cbrt_zzp() const { return def_.cbrt_zzp; }
    //! log(z)
    CELER_FUNCTION real_type log_z() const { return def_.log_z; }

    // Coulomb correction factor [unitless]
    inline CELER_FUNCTION real_type coulomb_correction() const;

    // Mass radiation coefficient for Bremsstrahlung [len^2/mass]
    inline CELER_FUNCTION real_type mass_radiation_coeff() const;

  private:
    MaterialParamsRef const& params_;
    ElementRecord const& def_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material data and global element ID.
 */
CELER_FUNCTION
ElementView::ElementView(MaterialParamsRef const& params, ElementId el_id)
    : params_(params), def_(params.elements[el_id])
{
    CELER_EXPECT(el_id < params.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Atomic number Z.
 *
 * Number of protons in an atom of this element.
 */
CELER_FUNCTION AtomicNumber ElementView::atomic_number() const
{
    return def_.atomic_number;
}

//---------------------------------------------------------------------------//
/*!
 * Number of isotopes available for this element.
 */
CELER_FUNCTION IsotopeComponentId::size_type ElementView::num_isotopes() const
{
    return def_.isotopes.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get isotope properties for a given index.
 */
CELER_FUNCTION IsotopeView ElementView::isotope_record(IsotopeComponentId id) const
{
    CELER_EXPECT(id < this->num_isotopes());
    return IsotopeView(params_, this->isotope_id(id));
}

//---------------------------------------------------------------------------//
/*!
 * ID of an isotope in this element.
 */
CELER_FUNCTION IsotopeId ElementView::isotope_id(IsotopeComponentId id) const
{
    CELER_EXPECT(id < this->num_isotopes());
    return this->isotopes()[id.get()].isotope;
}

//---------------------------------------------------------------------------//
/*!
 * View the isotopic components (id/fraction) of this element.
 */
CELER_FUNCTION Span<ElIsotopeComponent const> ElementView::isotopes() const
{
    return params_.isocomponents[def_.isotopes];
}

//---------------------------------------------------------------------------//
/*!
 * Coulomb correction term [unitless].
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
 * Mass radiation coefficient 1/X_0 for Bremsstrahlung [len^2/mass].
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
}  // namespace celeritas
