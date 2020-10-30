//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "ElementView.hh"
#include "MaterialParamsPointers.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access material properties.
 *
 * A material is a combination of nuclides/elements at a particular state (e.g.
 * density, temperature). The proportions and identities of a material's
 * constitutents are encoded in the \c elements accessor, where each index of
 * the returned span corresponds to an \c ElementComponentId for this material.
 * The \c get_element_density and \c element_view helper functions can be used
 * to calculate elemental densities and properties.
 *
 * \note The material -> nuclide mapping will be implemented when we add
 * hadronic physics. A separate NuclideComponentId and NuclideView will operate
 * analogously to the element access.
 */
class MaterialView
{
  public:
    // Construct from params and material ID
    inline CELER_FUNCTION
    MaterialView(const MaterialParamsPointers& params, MaterialDefId id);

    // >>> MATERIAL DATA

    // Number density [1/cm^3]
    inline CELER_FUNCTION real_type number_density() const;

    // Material temperature [K]
    inline CELER_FUNCTION real_type temperature() const;

    // Material state
    inline CELER_FUNCTION MatterState matter_state() const;

    // >>> ELEMENT ACCESS

    // Number of elemental components
    inline CELER_FUNCTION ElementComponentId::value_type num_elements() const;

    // Element properties for a material-specific index
    inline CELER_FUNCTION ElementView element_view(ElementComponentId id) const;

    // Total number density of an element in this material [1/cm^3]
    inline CELER_FUNCTION real_type
    get_element_density(ElementComponentId id) const;

    // Advanced access to the elemental components (id/fraction)
    inline CELER_FUNCTION const span<const MatElementComponent>&
                                elements() const;

    // >>> DERIVATIVE DATA

    // Material density [g/cm^3]
    inline CELER_FUNCTION real_type density() const;

    // Electrons per unit volume [1/cm^3]
    inline CELER_FUNCTION real_type electron_density() const;

    // Radiation length for high-energy electron Bremsstrahlung [cm]
    inline CELER_FUNCTION real_type radiation_length() const;

  private:
    const MaterialParamsPointers& params_;
    MaterialDefId                 id_;

    // HELPER FUNCTIONS

    inline CELER_FUNCTION const MaterialDef& material_def() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MaterialView.i.hh"
