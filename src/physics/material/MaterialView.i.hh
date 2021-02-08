//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic and static particle properties.
 */
CELER_FUNCTION
MaterialView::MaterialView(const MaterialParamsPointers& params, MaterialId id)
    : params_(params), material_(id)
{
    CELER_EXPECT(id < params.materials.size());
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
CELER_FUNCTION ElementComponentId::value_type MaterialView::num_elements() const
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
// PRIVATE METHODS
//---------------------------------------------------------------------------//
/*!
 * Static material defs for the current state.
 */
CELER_FUNCTION const MaterialDef& MaterialView::material_def() const
{
    return params_.materials[material_];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
