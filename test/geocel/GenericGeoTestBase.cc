//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestBase.hh"

#include "corecel/Config.hh"
#if CELERITAS_USE_GEANT4
#    include <G4LogicalVolume.hh>
#endif

#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoUtils.hh"

#include "CheckedGeoTrackView.hh"

using std::cout;
using namespace std::literals;
using GeantVolResult = celeritas::test::GenericGeoGeantImportVolumeResult;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void GenericGeoTrackingResult::print_expected()
{
    cout
        << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
           "static char const* const expected_volumes[] = "
        << repr(this->volumes)
        << ";\n"
           "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
           "static char const* const expected_volume_instances[] = "
        << repr(this->volume_instances)
        << ";\n"
           "EXPECT_VEC_EQ(expected_volume_instances, "
           "result.volume_instances);\n"
           "static real_type const expected_distances[] = "
        << repr(this->distances)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);\n"
           "static real_type const expected_hw_safety[] = "
        << repr(this->halfway_safeties)
        << ";\n"
           "EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);\n"
           "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
void GeantVolResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_volumes[] = "
         << repr(this->volumes)
         << ";\n"
            "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
            "EXPECT_EQ(0, result.missing_names.size()) << "
            "repr(result.missing_names);\n";
    if (!this->missing_names.empty())
    {
        cout << "/* Currently missing: " << repr(this->missing_names)
             << " */\n";
    }
    cout << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
GeantVolResult GeantVolResult::from_import(GeoParamsInterface const& geom,
                                           G4VPhysicalVolume const* world)
{
    CELER_VALIDATE(world, << "world volume is nullptr");

#if CELERITAS_USE_GEANT4
    using Result = GenericGeoGeantImportVolumeResult;

    auto vol_labels = make_logical_vol_labels(*world);

    Result result;
    result.volumes.resize(vol_labels.size());

    for (auto i : range(vol_labels.size()))
    {
        result.volumes[i] = [&label = vol_labels[i], &geom, &result] {
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
                result.missing_names.push_back(to_string(label));
                return Result::missing;
            }
        }();
    }

    // Trim leading 'empty' values
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
GeantVolResult GeantVolResult::from_pointers(GeoParamsInterface const& geom,
                                             G4VPhysicalVolume const* world)
{
    CELER_VALIDATE(world, << "world volume is nullptr");
#if CELERITAS_USE_GEANT4
    using Result = GenericGeoGeantImportVolumeResult;

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
            result.missing_names.push_back(lv->GetName());
        }
    }
    return result;
#else
    CELER_DISCARD(geom);
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
