//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantImportVolumeResult.cc
//---------------------------------------------------------------------------//
#include "GeantImportVolumeResult.hh"

#include <iostream>

#include "corecel/Config.hh"
#if CELERITAS_USE_GEANT4
#    include <G4LogicalVolume.hh>
#endif

#include "corecel/io/Repr.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"

#include "AssertionHelper.hh"
#include "Comparer.hh"
#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
GeantImportVolumeResult GeantImportVolumeResult::from_import(
    GeoParamsInterface const& geom)
{
#if CELERITAS_USE_GEANT4
    using Result = GeantImportVolumeResult;

    auto geant_geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");
    Result result;
    result.volumes.resize(geant_geo->impl_volumes().size());

    for (auto i : range<ImplVolumeId::size_type>(result.volumes.size()))
    {
        auto const& label = geant_geo->impl_volumes().at(ImplVolumeId{i});
        result.volumes[i] = [&] {
            if (label.empty())
            {
                return Result::empty;
            }
            else if (ImplVolumeId id = geom.impl_volumes().find_exact(label))
            {
                return static_cast<int>(id.get());
            }
            else
            {
                result.missing_labels.push_back(to_string(label));
                return Result::missing;
            }
        }();
    }

    // Trim leading 'empty' values
    // TODO: delete when doing LV/PV ID remapping
    auto first_nonempty = std::find_if(
        result.volumes.begin(), result.volumes.end(), [](int i) {
            return i != Result::empty;
        });
    result.volumes.erase(result.volumes.begin(), first_nonempty);

    return result;
#else
    CELER_DISCARD(geom);
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
std::ostream& operator<<(std::ostream& os, GeantImportVolumeResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, volumes) << ", "
              << CELER_ATTR_REPR(ref, missing_labels) << '}';
}

bool check_ref_eq(AssertionHelper& helper,
                  GeantImportVolumeResult const& expected,
                  GeantImportVolumeResult const& actual)
{
    RefComparer<GeantImportVolumeResult> compare{helper, expected, actual};
    CELER_INVOKE_ATTR(compare.is_eq, GeantImportVolumeResult, volumes);
    CELER_INVOKE_ATTR(compare.is_eq, GeantImportVolumeResult, missing_labels);
    return static_cast<bool>(compare);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
