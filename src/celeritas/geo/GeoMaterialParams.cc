//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/detail/Joined.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/Types.hh"
#include "celeritas/io/ImportData.hh"

#include "GeoMaterialData.hh"  // IWYU pragma: associated
#include "GeoParams.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
using MapLabelMatId = std::unordered_map<Label, PhysMatId>;

//---------------------------------------------------------------------------//
/*!
 * Construct a label -> material map from the input.
 *
 * The input is effectively an "unzipped" unordered list of (volume label,
 * material id) pairs.
 */
MapLabelMatId build_label_map(MaterialParams const& mat_params,
                              std::vector<Label>&& labels,
                              std::vector<PhysMatId> const& materials)
{
    CELER_EXPECT(materials.size() == labels.size());
    CELER_EXPECT(std::all_of(
        materials.begin(), materials.end(), [&mat_params](PhysMatId m) {
            return !m || m < mat_params.num_materials();
        }));

    std::unordered_map<Label, PhysMatId> lab_to_id;
    std::set<Label> duplicates;

    // Remap materials to volume IDs using given volume names:
    // build a map of volume name -> matid
    for (auto idx : range(materials.size()))
    {
        if (!materials[idx])
        {
            // Skip volumes without matids
            continue;
        }

        auto [prev, inserted]
            = lab_to_id.insert({std::move(labels[idx]), materials[idx]});
        if (!inserted)
        {
            duplicates.insert(prev->first);
        }
    }
    CELER_VALIDATE(duplicates.empty(),
                   << "geo/material coupling specified duplicate "
                      "volume names: \""
                   << join(duplicates.begin(), duplicates.end(), "\", \"")
                   << '"');
    return lab_to_id;
}

//---------------------------------------------------------------------------//
/*!
 * Find a material ID from a volume ID.
 */
class MaterialFinder
{
  public:
    MaterialFinder(GeoParams const& geo, MapLabelMatId const& materials)
        : geo_{geo}, materials_{materials}
    {
    }

    PhysMatId operator()(VolumeId const& volume_id)
    {
        Label const& vol_label = geo_.volumes().at(volume_id);

        // Hopefully user-provided and geo-provided volume labels match exactly
        if (auto iter = materials_.find(vol_label); iter != materials_.end())
        {
            return iter->second;
        }

        if (mat_labels_.empty())
        {
            // Build user volume -> material mapping
            this->build_mat_labels();
        }

        // Either:
        // - user-provided vol labels have no extensions? (just names)
        // - geometry volume labels are missing extensions (e.g. when
        //   Geant4-derived volume names, ORANGE geometry names)
        auto [start, stop] = mat_labels_.equal_range(vol_label.name);
        if (start == stop)
        {
            // No materials match the volume label
            return {};
        }

        std::set<PhysMatId> found_mat;
        for (auto iter = start; iter != stop; ++iter)
        {
            found_mat.insert(iter->second.second);
        }

        // Multiple labels with match with different materials
        if (found_mat.size() > 1)
        {
            CELER_LOG(warning)
                << "Multiple materials match the volume '" << vol_label
                << "': "
                << join_stream(
                       start, stop, ", ", [](std::ostream& os, auto&& mliter) {
                           os << mliter.second.first << "="
                              << mliter.second.second.unchecked_get();
                       });
        }
        return start->second.second;
    }

  private:
    GeoParams const& geo_;
    MapLabelMatId const& materials_;

    using PairExtMatid = std::pair<std::string, PhysMatId>;
    std::multimap<std::string, PairExtMatid> mat_labels_;

    void build_mat_labels()
    {
        for (auto const& [mlabel, matid] : materials_)
        {
            mat_labels_.emplace(mlabel.name, PairExtMatid{mlabel.ext, matid});
        }
    }
};

//---------------------------------------------------------------------------//
/*!
 * Whether a volume with a missing material needs to be reported to the user.
 */
bool ignore_volume_name(std::string const& name)
{
    if (name.empty())
        return true;
    if (name.front() == '[' && name.back() == ']')
        return true;
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a label -> material map from the input.
 */
std::vector<PhysMatId>
build_vol_to_mat(GeoParams const& geo, MapLabelMatId const& materials)
{
    auto const& vols = geo.volumes();
    std::vector<Label> missing_volumes;
    std::vector<PhysMatId> result(vols.size(), PhysMatId{});

    // Make sure at least one volume maps correctly
    VolumeId::size_type num_missing{0};

    // Map volume names to material names
    MaterialFinder find_matid{geo, materials};
    for (auto volume_id : range(VolumeId{vols.size()}))
    {
        if (auto matid = find_matid(volume_id))
        {
            result[volume_id.unchecked_get()] = matid;
            continue;
        }

        ++num_missing;
        Label const& label = vols.at(volume_id);
        if (!ignore_volume_name(label.name))
        {
            // Skip "[unused]" that we set for vecgeom empty labels,
            // "[EXTERIOR]" from ORANGE
            missing_volumes.push_back(label);
        }
    }

    if (!missing_volumes.empty())
    {
        CELER_LOG(warning)
            << "Some geometry volumes do not have known material IDs: "
            << join(missing_volumes.begin(), missing_volumes.end(), ", ");
    }

    auto mat_to_stream = [](std::ostream& os, auto& lab_mat) {
        os << '{' << lab_mat.first << ',';
        if (lab_mat.second)
        {
            os << lab_mat.second.unchecked_get();
        }
        else
        {
            os << '-';
        }
        os << '}';
    };

    // *ALL* volumes were absent
    CELER_VALIDATE(
        num_missing != vols.size(),
        << "no geometry volumes matched the available materials:\n"
           " materials: "
        << join_stream(materials.begin(), materials.end(), ", ", mat_to_stream)
        << "\n"
           "volumes: "
        << join(RangeIter<VolumeId>(VolumeId{0}),
                RangeIter<VolumeId>(VolumeId{vols.size()}),
                ", ",
                [&vols](VolumeId vid) { return vols.at(vid); }));

    // At least one material ID was assigned...
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<GeoMaterialParams>
GeoMaterialParams::from_import(ImportData const& data,
                               SPConstGeo geo_params,
                               SPConstMaterial material_params)
{
    GeoMaterialParams::Input input;
    input.geometry = std::move(geo_params);
    input.materials = std::move(material_params);

    input.volume_to_mat.resize(data.volumes.size());
    for (auto volume_idx :
         range<VolumeId::size_type>(input.volume_to_mat.size()))
    {
        if (!data.volumes[volume_idx])
            continue;

        input.volume_to_mat[volume_idx]
            = PhysMatId(data.volumes[volume_idx].phys_material_id);
    }

    // Assume that since Geant4 is using internal geometry and
    // we're using ORANGE or VecGeom that volume IDs will not be
    // the same. We'll just remap them based on their labels (which should
    // include uniquifying suffix if needed).
    input.volume_labels.resize(data.volumes.size());
    for (auto volume_idx : range(data.volumes.size()))
    {
        if (!data.volumes[volume_idx])
            continue;

        CELER_EXPECT(!data.volumes[volume_idx].name.empty());
        input.volume_labels[volume_idx]
            = Label::from_separator(data.volumes[volume_idx].name);
    }

    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and material params.
 *
 * Missing material IDs may be allowed if they correspond to unreachable volume
 * IDs.
 */
GeoMaterialParams::GeoMaterialParams(Input input)
{
    CELER_EXPECT(input.geometry);
    CELER_EXPECT(input.materials);
    CELER_EXPECT(
        (input.volume_labels.empty()
         && input.volume_to_mat.size() == input.geometry->volumes().size())
        || input.volume_to_mat.size() == input.volume_labels.size());

    ScopedMem record_mem("GeoMaterialParams.construct");

    if (!input.volume_labels.empty())
    {
        // User didn't provide an exact map of volume -> matid (typical case?)
        // Remap based on labels
        auto lab_to_id = build_label_map(*input.materials,
                                         std::move(input.volume_labels),
                                         input.volume_to_mat);

        // Reconstruct volume-to-material mapping from label map and geometry
        input.volume_to_mat
            = build_vol_to_mat(*input.geometry, std::move(lab_to_id));
    }
    CELER_ASSERT(input.volume_to_mat.size()
                 == input.geometry->volumes().size());

    HostValue host_data;
    auto materials = make_builder(&host_data.materials);
    materials.insert_back(input.volume_to_mat.begin(),
                          input.volume_to_mat.end());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GeoMaterialParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
