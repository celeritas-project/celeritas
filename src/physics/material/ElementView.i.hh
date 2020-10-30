//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ElementView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared material pointers and global element ID.
 */
CELER_FUNCTION
ElementView::ElementView(const MaterialParamsPointers& params,
                         ElementDefId                  el_id)
    : def_(params.elements[el_id.get()])
{
    REQUIRE(el_id < params.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Mass radiation coefficient for Bremsstrahlung [cm^2/g].
 *
 * The "radiation length" X_0 is "the mean distance over which a high-energy
 * electron loses all but 1/e of its energy by bremsstrahlung".  [section
 * 33.4.2 (p452) of "Review of Particle Physics" (PDG group), Phys.  Rev. D 98
 * .]
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
