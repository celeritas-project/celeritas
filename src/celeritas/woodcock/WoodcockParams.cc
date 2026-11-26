//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockParams.cc
//---------------------------------------------------------------------------//
#include "WoodcockParams.hh"

#include <map>
#include <set>
#include <string>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/VolumeVisitor.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTraits.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/grid/UniformGridInserter.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/MajorantBuilder.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Find all the volumes that belong to the selected regions.
 *
 * A single user-defined region can be split into several regions that share a
 * label name, with the extension identifying a volume with overridden data.
 */
std::set<VolumeId> find_volumes(std::vector<std::string> const& names,
                                std::vector<inp::Region> const& regions)
{
    std::set<VolumeId> result;
    for (std::string const& name : names)
    {
        bool found{false};
        for (inp::Region const& r : regions)
        {
            if (r.label.name == name)
            {
                found = true;
                result.insert(r.volumes.begin(), r.volumes.end());
            }
        }
        if (!found)
        {
            std::set<std::string> available;
            for (inp::Region const& r : regions)
            {
                available.insert(r.label.name);
            }
            CELER_VALIDATE(found,
                           << "Woodcock tracking region '" << name
                           << "' does not exist or contains no volumes "
                              "(available: '"
                           << join(available.begin(), available.end(), "', '")
                           << "')");
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find selected volumes that are the world or are placed inside an unselected
 * volume.
 */
std::set<VolumeId> find_roots(std::set<VolumeId> const& selected,
                              VolumeParams const& volumes)
{
    std::set<VolumeId> result;
    if (selected.count(volumes.world()))
    {
        result.insert(volumes.world());
    }
    for (auto mother : range(VolumeId{volumes.num_volumes()}))
    {
        if (selected.count(mother))
        {
            continue;
        }
        for (auto vi : volumes.children(mother))
        {
            if (VolumeId vol = volumes.volume(vi); selected.count(vol))
            {
                result.insert(vol);
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Map canonical volumes to all the physics materials they contain.
 */
std::vector<std::set<PhysMatId>> make_volume_materials(
    CoreGeoParams const& geo,
    GeoMaterialParams const& geomaterial,
    VolumeParams const& volumes)
{
    std::vector<std::set<PhysMatId>> result(volumes.num_volumes());
    for (auto iv_id : range(ImplVolumeId{geo.impl_volumes().size()}))
    {
        VolumeId vol = geo.volume_id(iv_id);
        PhysMatId mat = geomaterial.material_id(iv_id);
        if (vol && mat)
        {
            CELER_ASSERT(vol < result.size());
            result[vol.get()].insert(mat);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from options and region definitions.
 */
WoodcockParams::WoodcockParams(Input const& input,
                               inp::WoodcockTracking const& options,
                               inp::Regions const& regions)
{
    CELER_EXPECT(input);
    CELER_EXPECT(options);

    CELER_VALIDATE(CoreGeoTraits::has_ancestor_exit,
                   << "Woodcock tracking is not implemented for this version "
                      "of "
                   << CoreGeoTraits::name << " geometry");
    CELER_VALIDATE(options.min_energy > zero_quantity(),
                   << "nonpositive Woodcock tracking minimum energy");
    CELER_VALIDATE(options.max_flights > 0,
                   << "nonpositive Woodcock tracking flight limit");
    CELER_VALIDATE(!input.volume->empty(),
                   << "Woodcock tracking requires volume structure data");

    HostVal<WoodcockParamsData> data;

    // Set scalars
    data.scalars.gamma = input.particle->find(pdg::gamma());
    CELER_VALIDATE(data.scalars.gamma,
                   << "Woodcock tracking requires gammas to be available");
    detail::MajorantBuilder build_majorant(*input.material,
                                           *input.particle,
                                           *input.physics,
                                           data.scalars.gamma,
                                           options.min_energy);
    data.scalars.min_energy = options.min_energy;
    data.scalars.max_energy = build_majorant.max_energy();
    data.scalars.max_flights = options.max_flights;

    // Find root volumes and everything inside
    auto const& volumes = *input.volume;
    auto const roots
        = find_roots(find_volumes(options.regions, regions.regions), volumes);
    CELER_VALIDATE(!roots.empty(),
                   << "no volumes are available for Woodcock tracking");
    auto const vol_to_mat
        = make_volume_materials(*input.geometry, *input.geomaterial, volumes);

    std::vector<WoodcockRootId> vol_to_root(volumes.num_volumes());
    std::vector<unsigned char> inside(volumes.num_volumes(), 0);
    CollectionBuilder majorants{&data.majorants};
    Collection<UniformGridRecord, Ownership::value, MemSpace::host> grids;
    UniformGridInserter insert_grid(&data.reals, &grids);
    std::map<std::set<PhysMatId>, UniformGridRecord> mat_to_grid;

    for (VolumeId root : roots)
    {
        std::set<PhysMatId> materials;
        VolumeVisitor{volumes}(
            make_visit_volume_once<VolumeId>([&](VolumeId vol) {
                inside[vol.get()] = 1;
                auto const& m = vol_to_mat[vol.get()];
                materials.insert(m.begin(), m.end());
            }),
            root);

        // Build majorant, or reuse if the materials are the same
        auto [iter, inserted] = mat_to_grid.insert({materials, {}});
        if (inserted)
        {
            auto grid_id = insert_grid(
                build_majorant({materials.begin(), materials.end()}));
            iter->second = grids[grid_id];
        }
        vol_to_root[root.get()] = majorants.push_back(iter->second);
    }

    make_builder(&data.roots).insert_back(vol_to_root.begin(), vol_to_root.end());
    make_builder(&data.inside).insert_back(inside.begin(), inside.end());

    CELER_LOG(info) << "Using Woodcock tracking for gammas in " << roots.size()
                    << " root volume" << (roots.size() == 1 ? "" : "s")
                    << " with " << mat_to_grid.size()
                    << " majorant cross section"
                    << (mat_to_grid.size() == 1 ? "" : "s");

    CELER_ASSERT(data);
    data_ = ParamsDataStore<WoodcockParamsData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
