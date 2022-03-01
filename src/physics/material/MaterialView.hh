//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

#include "ElementView.hh"
#include "MaterialData.hh"
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
    //!@{
    //! Type aliases
    using MaterialParamsRef
        = MaterialParamsData<Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from params and material ID
    inline CELER_FUNCTION
    MaterialView(const MaterialParamsRef& params, MaterialId id);

    //// MATERIAL DATA ////

    // ID of this Material
    CELER_FORCEINLINE_FUNCTION MaterialId material_id() const;

    // Number density [1/cm^3]
    CELER_FORCEINLINE_FUNCTION real_type number_density() const;

    // Material temperature [K]
    CELER_FORCEINLINE_FUNCTION real_type temperature() const;

    // Material state
    CELER_FORCEINLINE_FUNCTION MatterState matter_state() const;

    //// ELEMENT ACCESS ////

    // Number of elemental components
    inline CELER_FUNCTION ElementComponentId::size_type num_elements() const;

    // Element properties for a material-specific index
    inline CELER_FUNCTION ElementView element_view(ElementComponentId id) const;

    // ID of a component element in this material
    inline CELER_FUNCTION ElementId element_id(ElementComponentId id) const;

    // Total number density of an element in this material [1/cm^3]
    inline CELER_FUNCTION real_type
                          get_element_density(ElementComponentId id) const;

    // Advanced access to the elemental components (id/fraction)
    inline CELER_FUNCTION Span<const MatElementComponent> elements() const;

    //// DERIVATIVE DATA ////

    // Material density [g/cm^3]
    inline CELER_FUNCTION real_type density() const;

    // Electrons per unit volume [1/cm^3]
    inline CELER_FUNCTION real_type electron_density() const;

    // Radiation length for high-energy electron Bremsstrahlung [cm]
    inline CELER_FUNCTION real_type radiation_length() const;

    // Mean excitation energy [MeV]
    inline CELER_FUNCTION units::MevEnergy mean_excitation_energy() const;

    // Log mean excitation energy
    inline CELER_FUNCTION units::LogMevEnergy
                          log_mean_excitation_energy() const;

  private:
    const MaterialParamsRef& params_;
    MaterialId               material_;

    // HELPER FUNCTIONS

    CELER_FORCEINLINE_FUNCTION const MaterialRecord& material_def() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
MaterialView::MaterialView(const MaterialParamsRef& params, MaterialId id)
    : params_(params), material_(id)
{
    CELER_EXPECT(id < params.materials.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get material id
 */
CELER_FUNCTION MaterialId MaterialView::material_id() const
{
    return material_;
}

//---------------------------------------------------------------------------//
/*!
 * Get atomic number density [1/cm^3].
 */
CELER_FUNCTION real_type MaterialView::number_density() const
{
    return this->material_def().number_density;
}

//---------------------------------------------------------------------------//
/*!
 * Get material temperature [K].
 */
CELER_FUNCTION real_type MaterialView::temperature() const
{
    return this->material_def().temperature;
}

//---------------------------------------------------------------------------//
/*!
 * Get material's state of matter (gas, liquid, solid).
 */
CELER_FUNCTION MatterState MaterialView::matter_state() const
{
    return this->material_def().matter_state;
}

//---------------------------------------------------------------------------//
/*!
 * Number of elements present in this material.
 */
CELER_FUNCTION ElementComponentId::size_type MaterialView::num_elements() const
{
    return this->material_def().elements.size();
}

//---------------------------------------------------------------------------//
/*!
 * Get element properties from a material-specific index.
 */
CELER_FUNCTION ElementView MaterialView::element_view(ElementComponentId id) const
{
    CELER_EXPECT(id < this->material_def().elements.size());
    return ElementView(params_, this->element_id(id));
}

//---------------------------------------------------------------------------//
/*!
 * ID of a component element in this material.
 */
CELER_FUNCTION ElementId MaterialView::element_id(ElementComponentId id) const
{
    CELER_EXPECT(id < this->material_def().elements.size());
    return this->elements()[id.get()].element;
}

//---------------------------------------------------------------------------//
/*!
 * Number density of an element in this material [1/cm^3]
 */
CELER_FUNCTION real_type
               MaterialView::get_element_density(ElementComponentId id) const
{
    CELER_EXPECT(id < this->material_def().elements.size());
    return this->number_density() * this->elements()[id.get()].fraction;
}

//---------------------------------------------------------------------------//
/*!
 * View the elemental components (id/fraction) of this material.
 */
CELER_FUNCTION Span<const MatElementComponent> MaterialView::elements() const
{
    return params_.elcomponents[this->material_def().elements];
}

//---------------------------------------------------------------------------//
/*!
 * Material's density [g/cm^3].
 */
CELER_FUNCTION real_type MaterialView::density() const
{
    return this->material_def().density;
}

//---------------------------------------------------------------------------//
/*!
 * Electrons per unit volume [1/cm^3].
 */
CELER_FUNCTION real_type MaterialView::electron_density() const
{
    return this->material_def().electron_density;
}

//---------------------------------------------------------------------------//
/*!
 * Radiation length for high-energy electron Bremsstrahlung [cm].
 */
CELER_FUNCTION real_type MaterialView::radiation_length() const
{
    return this->material_def().rad_length;
}

//---------------------------------------------------------------------------//
/*!
 * Mean excitation energy [MeV].
 */
CELER_FUNCTION units::MevEnergy MaterialView::mean_excitation_energy() const
{
    return this->material_def().mean_exc_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Log mean excitation energy.
 */
CELER_FUNCTION units::LogMevEnergy
               MaterialView::log_mean_excitation_energy() const
{
    return this->material_def().log_mean_exc_energy;
}

//---------------------------------------------------------------------------//
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Static material defs for the current state.
 */
CELER_FUNCTION const MaterialRecord& MaterialView::material_def() const
{
    return params_.materials[material_];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
