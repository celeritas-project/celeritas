//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/detail/Joined.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "CoreGeoParams.hh"  // IWYU pragma: keep
#include "GeoMaterialData.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace
{
using VecMat = std::vector<PhysMatId>;
using ImplVolumeMap = GeoParamsInterface::ImplVolumeMap;

//---------------------------------------------------------------------------//
/*!
 * Find a material ID from a volume ID.
 */
class MaterialFinder
{
  public:
    using MapLabelMat = GeoMaterialParams::Input::MapLabelMat;

    MaterialFinder(ImplVolumeMap const& labels, MapLabelMat const& materials)
        : iv_labels_{labels}, materials_{materials}
    {
    }

    PhysMatId operator()(ImplVolumeId impl_id)
    {
        CELER_EXPECT(impl_id < iv_labels_.size());
        Label const& vol_label = iv_labels_.at(impl_id);

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
    ImplVolumeMap const& iv_labels_;
    MapLabelMat const& materials_;

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
 * Construct physics materials for each ImplVolume from VolumeParams volumes.
 */
struct BuildFromCanonicalVolumes
{
    CoreGeoParams const& geo;

    VecMat operator()(VecMat const& materials) const;
};

VecMat BuildFromCanonicalVolumes::operator()(VecMat const& materials) const
{
    CELER_LOG(debug) << "Filling geometry->physics map using canonical "
                        "volumes";

    // Loop over implementation volumes, querying for the corresponding
    // canonical volume
    VecMat result(geo.impl_volumes().size());
    for (auto impl_id : range(id_cast<ImplVolumeId>(result.size())))
    {
        if (auto vol_id = geo.volume_id(impl_id))
        {
            CELER_ASSERT(vol_id < materials.size());
            result[impl_id.get()] = materials[vol_id.get()];
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a label -> material map from the input.
 */
struct BuildFromLabels
{
    using MapLabelMat = GeoMaterialParams::Input::MapLabelMat;

    ImplVolumeMap const& iv_labels;

    VecMat operator()(MapLabelMat const& materials) const;
};

VecMat BuildFromLabels::operator()(MapLabelMat const& materials) const
{
    CELER_LOG(debug) << "Filling geometry->physics map using label map";

    std::vector<std::reference_wrapper<Label const>> missing;
    std::vector<PhysMatId> result(iv_labels.size(), PhysMatId{});

    // Make sure at least one volume maps correctly
    ImplVolumeId::size_type num_missing{0};

    // Map volume names to material names
    MaterialFinder find_matid{iv_labels, materials};
    for (auto impl_id : range(ImplVolumeId{iv_labels.size()}))
    {
        if (auto matid = find_matid(impl_id))
        {
            result[impl_id.unchecked_get()] = matid;
            continue;
        }

        ++num_missing;
        Label const& label = iv_labels.at(impl_id);
        if (!ignore_volume_name(label.name))
        {
            // Skip "[unused]" that we set for vecgeom empty labels,
            // "[EXTERIOR]" from ORANGE
            missing.emplace_back(label);
        }
    }

    if (!missing.empty())
    {
        CELER_LOG(warning)
            << "Some geometry volumes do not have known material IDs: "
            << join(missing.begin(), missing.end(), ", ");
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
        num_missing != iv_labels.size(),
        << "no geometry volumes matched the available materials:\n"
           " materials: "
        << join_stream(materials.begin(), materials.end(), ", ", mat_to_stream)
        << "\n"
           "volumes: "
        << join(RangeIter<ImplVolumeId>(ImplVolumeId{0}),
                RangeIter<ImplVolumeId>(ImplVolumeId{iv_labels.size()}),
                ", ",
                [this](ImplVolumeId vid) { return iv_labels.at(vid); }));

    // At least one material ID was assigned...
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a sparse map of implementation volume IDs.
 */
struct BuildFromImplVolumes
{
    using MapImplMat = GeoMaterialParams::Input::MapImplMat;

    ImplVolumeMap const& iv_labels;

    VecMat operator()(MapImplMat const& materials) const;
};

VecMat BuildFromImplVolumes::operator()(MapImplMat const& materials) const
{
    CELER_LOG(debug) << "Filling geometry->physics map using ImplVolumeId map";

    // Loop over implementation volumes, querying for the corresponding
    // canonical volume
    VecMat result(iv_labels.size());

    std::vector<std::reference_wrapper<Label const>> missing;

    for (auto impl_id : range(id_cast<ImplVolumeId>(result.size())))
    {
        auto iter = materials.find(impl_id);
        if (iter != materials.end())
        {
            auto matid = iter->second;
            CELER_EXPECT(matid);
            result[impl_id.get()] = matid;
        }
        else
        {
            Label const& label = iv_labels.at(impl_id);

            if (!ignore_volume_name(label.name))
            {
                missing.emplace_back(label);
            }
        }
    }

    if (!missing.empty())
    {
        CELER_LOG(warning)
            << "Some geometry volumes do not have known material IDs: "
            << join(missing.begin(), missing.end(), ", ");
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 *
 * Note that the import volume index (see GeantImporter.cc) corresponds to the
 * canonical \c VolumeId .
 */
std::shared_ptr<GeoMaterialParams>
GeoMaterialParams::from_import(ImportData const& data,
                               SPConstCoreGeo geo_params,
                               SPConstVolume vol_params,
                               SPConstMaterial material_params)
{
    GeoMaterialParams::Input input;

    if (vol_params && !vol_params->empty())
    {
        // Construct vector of material IDs for each canonical volume
        CELER_LOG(debug)
            << "Building geometry->physics map using VolumeParams ("
            << vol_params->num_volumes() << " volumes)";
        Input::VecMat vol_to_mat(data.volumes.size());
        for (auto vol_idx : range(data.volumes.size()))
        {
            auto const& inp_vol = data.volumes[vol_idx];
            if (!inp_vol)
                continue;

            vol_to_mat[vol_idx] = PhysMatId(inp_vol.phys_material_id);
        }
        input.volume_to_mat = std::move(vol_to_mat);
    }
    else
    {
        // No volume information available: remap based on labels (which should
        // include uniquifying suffix if needed).
        CELER_ASSERT(geo_params);
        CELER_LOG(debug) << "Building geometry->physics map using labels ("
                         << geo_params->impl_volumes().size()
                         << " impl volumes)";

        Input::MapLabelMat label_to_mat;
        for (auto vol_idx : range(data.volumes.size()))
        {
            auto const& inp_vol = data.volumes[vol_idx];
            if (!inp_vol)
                continue;

            CELER_EXPECT(!inp_vol.name.empty());
            auto&& [iter, inserted] = label_to_mat.emplace(
                Label::from_separator(inp_vol.name), inp_vol.phys_material_id);
            if (!inserted)
            {
                CELER_LOG(error)
                    << "Duplicate volume input label '" << inp_vol.name << "'";
            }
        }
        input.volume_to_mat = std::move(label_to_mat);
    }

    input.geometry = std::move(geo_params);
    input.materials = std::move(material_params);
    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and material params.
 *
 * Missing material IDs may be allowed if they correspond to unreachable volume
 * IDs.
 */
GeoMaterialParams::GeoMaterialParams(Input const& input)
{
    CELER_EXPECT(input.geometry);
    CELER_EXPECT(input.materials);

    ScopedMem record_mem("GeoMaterialParams.construct");

    auto const& impl_volumes = input.geometry->impl_volumes();
    VecMat volume_to_mat = std::visit(
        Overload{
            // Canonical ID input
            BuildFromCanonicalVolumes{*input.geometry},
            // Build from labels
            BuildFromLabels{impl_volumes},
            // Build from a vector of implementation volumes
            BuildFromImplVolumes{impl_volumes},
        },
        input.volume_to_mat);
    CELER_ASSERT(volume_to_mat.size() == impl_volumes.size());

    HostValue host_data;
    CollectionBuilder(&host_data.materials)
        .insert_back(volume_to_mat.begin(), volume_to_mat.end());
    CELER_ASSERT(host_data);

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GeoMaterialParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
