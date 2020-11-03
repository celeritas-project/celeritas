//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "base/DeviceVector.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "ElementDef.hh"
#include "MaterialDef.hh"
#include "Types.hh"
#include "MaterialParamsPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data management for material, element, and nuclide properties.
 */
class MaterialParams
{
  public:
    //@{
    //! Type aliases
    //@}

    struct ElementInput
    {
        int            atomic_number; //!< Z number
        units::AmuMass atomic_mass;   //!< Isotope-weighted average atomic mass
        std::string    name;          //!< Element name
    };

    struct MaterialInput
    {
        real_type   number_density; //!< Atomic number density [1/cm^3]
        real_type   temperature;    //!< Temperature [K]
        MatterState matter_state;   //!< Solid, liquid, gas
        std::vector<std::pair<ElementDefId, real_type>> elements;

        std::string name;
    };

    struct Input
    {
        std::vector<ElementInput>  elements;
        std::vector<MaterialInput> materials;
    };

  public:
    // Construct with a vector of material definitions
    explicit MaterialParams(const Input& inp);

    // Get element name
    inline const std::string& id_to_label(ElementDefId id) const;

    // Get material name
    inline const std::string& id_to_label(MaterialDefId id) const;

    // Find a material from a name
    inline MaterialDefId find(const std::string& name) const;

    // Access material properties on the host
    MaterialParamsPointers host_pointers() const;

    // Access material properties on the device
    MaterialParamsPointers device_pointers() const;

    //! Maximum number of elements in any one material
    size_type max_element_components() const { return max_el_; }

  private:
    std::vector<ElementDef>          host_elements_;
    std::vector<MatElementComponent> host_elcomponents_;
    std::vector<MaterialDef>         host_materials_;

    DeviceVector<ElementDef>          device_elements_;
    DeviceVector<MatElementComponent> device_elcomponents_;
    DeviceVector<MaterialDef>         device_materials_;

    std::vector<std::string>                       elnames_;
    std::vector<std::string>                       matnames_;
    std::unordered_map<std::string, MaterialDefId> matname_to_id_;
    size_type                                      max_el_;

    // HELPER FUNCTIONS
    void                      append_element_def(const ElementInput& el);
    span<MatElementComponent> extend_elcomponents(const MaterialInput& el);
    void                      append_material_def(const MaterialInput& el);
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MaterialParams.i.hh"
