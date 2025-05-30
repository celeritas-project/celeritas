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

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
GeantImportVolumeResult
GeantImportVolumeResult::from_import(GeoParamsInterface const& geom)
{
#if CELERITAS_USE_GEANT4
    using Result = GeantImportVolumeResult;

    auto* geant_geo = celeritas::geant_geo();
    CELER_VALIDATE(geant_geo, << "global Geant4 geometry is not loaded");
    Result result;
    result.volumes.resize(geant_geo->volumes().size());

    for (auto i : range<VolumeId::size_type>(result.volumes.size()))
    {
        auto const& label = geant_geo->volumes().at(VolumeId{i});
        result.volumes[i] = [&] {
            if (label.empty())
            {
                return Result::empty;
            }
            else if (VolumeId id = geom.volumes().find_exact(label))
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
GeantImportVolumeResult
GeantImportVolumeResult::from_pointers(GeoParamsInterface const& geom)
{
#if CELERITAS_USE_GEANT4
    using Result = GeantImportVolumeResult;

    Result result;
    for (G4LogicalVolume* lv : celeritas::geant_logical_volumes())
    {
        if (!lv)
        {
            result.volumes.push_back(Result::empty);
            continue;
        }
        auto id = geom.find_volume(lv);
        result.volumes.push_back(id ? static_cast<int>(id.unchecked_get())
                                    : Result::missing);
        if (!id)
        {
            result.missing_labels.push_back(lv->GetName());
        }
    }
    return result;
#else
    CELER_DISCARD(geom);
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
void GeantImportVolumeResult::print_expected() const
{
    using std::cout;

    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_volumes[] = "
         << repr(this->volumes)
         << ";\n"
            "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
            "EXPECT_EQ(0, result.missing_labels.size()) << "
            "repr(result.missing_labels);\n";
    if (!this->missing_labels.empty())
    {
        cout << "/* Currently missing: " << repr(this->missing_labels)
             << " */\n";
    }
    cout << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
