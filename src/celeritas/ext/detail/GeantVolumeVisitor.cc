//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantVolumeVisitor.cc
//---------------------------------------------------------------------------//
#include "GeantVolumeVisitor.hh"

#include <G4GDMLWriteStructure.hh>
#include <G4LogicalVolume.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ReflectionFactory.hh>
#include <G4VSolid.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/io/ImportVolume.hh"

#include "../GeantGeoUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate the GDML name for a Geant4 logical volume.
 */
std::string
GeantVolumeVisitor::generate_name(G4LogicalVolume const& logical_volume)
{
    return GeantVolumeVisitor::generate_name_refl(logical_volume).first;
}

//---------------------------------------------------------------------------//
/*!
 * Store all logical volumes by recursively looping over them.
 *
 * Using a map ensures that volumes are both ordered by volume id and not
 * duplicated.
 * Function called by \c store_volumes(...) .
 */
void GeantVolumeVisitor::visit(G4LogicalVolume const& logical_volume)
{
    auto&& [iter, inserted] = volids_volumes_.emplace(
        logical_volume.GetInstanceID(), ImportVolume{});
    if (!inserted)
    {
        // Logical volume is already in the map
        return;
    }

    CELER_ASSERT(iter->first >= 0);

    // Fill volume properties
    ImportVolume& volume = iter->second;

    if (auto* cuts = logical_volume.GetMaterialCutsCouple())
    {
        volume.material_id = cuts->GetIndex();
    }
    volume.name = logical_volume.GetName();
    volume.solid_name = logical_volume.GetSolid()->GetName();

    if (volume.name.empty())
    {
        CELER_LOG(warning)
            << "No logical volume name specified for instance ID "
            << iter->first << " (material " << volume.material_id << ")";
    }
    else if (unique_volumes_)
    {
        volume.name = this->generate_name(logical_volume);
    }

    // Recursive: repeat for every daughter volume, if there are any
    for (auto const i : range(logical_volume.GetNoDaughters()))
    {
        this->visit(*logical_volume.GetDaughter(i)->GetLogicalVolume());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Transform the map of volumes into a contiguous vector with empty spaces.
 */
std::vector<ImportVolume> GeantVolumeVisitor::build_volume_vector() const
{
    std::vector<ImportVolume> volumes;

    // Populate vector<ImportVolume>
    volumes.resize(volids_volumes_.size());
    for (auto&& [volid, volume] : volids_volumes_)
    {
        if (static_cast<std::size_t>(volid) >= volumes.size())
        {
            volumes.resize(volid + 1);
        }
        volumes[volid] = volume;
    }

    return volumes;
}

//---------------------------------------------------------------------------//
/*!
 * Transform the map of volumes into a list of labels.
 *
 * This is used by GeantGeoParams.
 */
std::vector<Label> GeantVolumeVisitor::build_label_vector() const
{
    std::vector<Label> labels;

    // Populate vector<ImportVolume>
    labels.resize(volids_volumes_.size());
    for (auto&& [volid, volume] : volids_volumes_)
    {
        if (static_cast<std::size_t>(volid) >= labels.size())
        {
            labels.resize(volid + 1);
        }
        labels[volid] = Label::from_geant(volume.name);
    }

    return labels;
}

//---------------------------------------------------------------------------//
/*!
 * Generate the GDML name and return a pointer to an underlying volume.
 */
std::pair<std::string, G4LogicalVolume const*>
GeantVolumeVisitor::generate_name_refl(G4LogicalVolume const& logical_volume)
{
    // Run the LV through the GDML export name generator so that the volume is
    // uniquely identifiable in VecGeom. Reuse the same instance to reduce
    // overhead: note that the method isn't const correct.
    static G4GDMLWriteStructure temp_writer;

    auto const* refl_factory = G4ReflectionFactory::Instance();
    if (auto const* unrefl_lv = refl_factory->GetConstituentLV(
            const_cast<G4LogicalVolume*>(&logical_volume)))
    {
        // If this is a reflected volume, add the reflection extension after
        // the final pointer to match the converted VecGeom name
        std::string name
            = temp_writer.GenerateName(unrefl_lv->GetName(), unrefl_lv);
        name += refl_factory->GetVolumesNameExtension();
        return {std::move(name), unrefl_lv};
    }

    std::string name
        = temp_writer.GenerateName(logical_volume.GetName(), &logical_volume);
    return {std::move(name), nullptr};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
