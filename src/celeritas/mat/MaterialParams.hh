//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "ElementView.hh"
#include "IsotopeView.hh"
#include "MaterialData.hh"
#include "MaterialView.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Manage material, element, and nuclide properties.
 *
 * Materials in Celeritas currently correspond to "material cut couples" in
 * Geant4, i.e. the outer product of geometry model-defined materials and
 * user-defined physics regions.
 *
 * \todo Replace id_to_label etc. with direct access to LabelIdMultiMap
 * \todo Split into isotope/element/geo material
 */
class MaterialParams final : public ParamsDataInterface<MaterialParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SpanConstMaterialId = Span<MaterialId const>;
    using SpanConstElementId = Span<ElementId const>;
    using SpanConstIsotopeId = Span<IsotopeId const>;
    //!@}

    //! Define an element's isotope input data
    struct IsotopeInput
    {
        //!@{
        //! \name Type aliases
        using AtomicMassNumber = AtomicNumber;
        using MevEnergy = units::MevEnergy;
        //!@}

        AtomicNumber atomic_number;  //!< Atomic number Z
        AtomicMassNumber atomic_mass_number;  //!< Atomic number A
        MevEnergy binding_energy;  //!< Nuclear binding energy (BE)
        MevEnergy proton_loss_energy;  //!< BE(A, Z) - BE(A-1, Z-1)
        MevEnergy neutron_loss_energy;  //!< BE(A, Z) - BE(A-1, Z)
        units::MevMass nuclear_mass;  //!< Nucleons' mass + binding energy
        Label label;  //!< Isotope name
    };

    //! Define an element's input data
    struct ElementInput
    {
        AtomicNumber atomic_number;  //!< Atomic number Z
        units::AmuMass atomic_mass;  //!< Isotope-weighted average atomic mass
        std::vector<std::pair<IsotopeId, real_type>>
            isotopes_fractions;  //!< Isotopic fractional abundance
        Label label;  //!< Element name
    };

    //! Define a material's input data
    struct MaterialInput
    {
        real_type number_density;  //!< Atomic number density [1/length^3]
        real_type temperature;  //!< Temperature [K]
        MatterState matter_state;  //!< Solid, liquid, gas
        std::vector<std::pair<ElementId, real_type>>
            elements_fractions;  //!< Fraction of number density
        Label label;  //!< Material name
    };

    //! Input data to construct this class
    struct Input
    {
        std::vector<IsotopeInput> isotopes;
        std::vector<ElementInput> elements;
        std::vector<MaterialInput> materials;
        std::vector<OpticalMaterialId> mat_to_optical;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<MaterialParams> from_import(ImportData const& data);

    // Construct with a vector of material definitions
    explicit MaterialParams(Input const& inp);

    //! Number of material definitions
    MaterialId::size_type size() const { return mat_labels_.size(); }

    //!@{
    //! \name Material metadata

    //! Number of materials
    MaterialId::size_type num_materials() const { return mat_labels_.size(); }

    // Get material name
    Label const& id_to_label(MaterialId id) const;

    // Find a material from a name
    MaterialId find_material(std::string const& name) const;

    // Find all materials that share a name
    SpanConstMaterialId find_materials(std::string const& name) const;
    //!@}

    //!@{
    //! \name Element metadata

    //! Number of distinct elements definitions
    ElementId::size_type num_elements() const { return el_labels_.size(); }

    // Get element name
    Label const& id_to_label(ElementId id) const;

    // Find an element from a name
    ElementId find_element(std::string const& name) const;

    // Find all elements that share a name
    SpanConstElementId find_elements(std::string const& name) const;
    //!@}

    //!@{
    //! \name Isotope metadata

    //! Number of distinct isotope definitions
    IsotopeId::size_type num_isotopes() const { return isot_labels_.size(); }

    // Get isotope name
    Label const& id_to_label(IsotopeId id) const;

    // Find an isotope from a name
    IsotopeId find_isotope(std::string const& name) const;

    // Find all isotopes that share a name
    SpanConstIsotopeId find_isotopes(std::string const& name) const;
    //!@}

    // Access material definitions on host
    inline MaterialView get(MaterialId id) const;

    // Access element definitions on host
    inline ElementView get(ElementId id) const;

    // Access isotope definitions on host
    inline IsotopeView get(IsotopeId id) const;

    // Maximum number of isotopes in any one element
    inline IsotopeComponentId::size_type max_isotope_components() const;

    // Maximum number of elements in any one material
    inline ElementComponentId::size_type max_element_components() const;

    //! Whether isotopic data is present (may be false for EM-only physics)
    inline bool is_missing_isotopes() const { return this->num_isotopes(); }

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Metadata
    LabelIdMultiMap<MaterialId> mat_labels_;
    LabelIdMultiMap<ElementId> el_labels_;
    LabelIdMultiMap<IsotopeId> isot_labels_;

    // Host/device storage and reference
    CollectionMirror<MaterialParamsData> data_;

    // HELPER FUNCTIONS
    using HostValue = HostVal<MaterialParamsData>;
    void append_element_def(ElementInput const& inp, HostValue*);
    void append_isotope_def(IsotopeInput const& inp, HostValue*);
    ItemRange<MatElementComponent>
    extend_elcomponents(MaterialInput const& inp, HostValue*) const;
    void append_material_def(MaterialInput const& inp, HostValue*);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get material properties for the given material.
 */
MaterialView MaterialParams::get(MaterialId id) const
{
    CELER_EXPECT(id < this->host_ref().materials.size());
    return MaterialView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Get properties for the given element.
 */
ElementView MaterialParams::get(ElementId id) const
{
    CELER_EXPECT(id < this->host_ref().elements.size());
    return ElementView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Get properties for the given isotope.
 */
IsotopeView MaterialParams::get(IsotopeId id) const
{
    CELER_EXPECT(id < this->host_ref().isotopes.size());
    return IsotopeView(this->host_ref(), id);
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of isotopes in any one element.
 */
IsotopeComponentId::size_type MaterialParams::max_isotope_components() const
{
    return this->host_ref().max_isotope_components;
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of elements in any one material.
 */
ElementComponentId::size_type MaterialParams::max_element_components() const
{
    return this->host_ref().max_element_components;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
