//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
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
