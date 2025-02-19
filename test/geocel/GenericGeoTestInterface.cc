//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include <gtest/gtest.h>

#include "corecel/Config.hh"

#include "corecel/io/Repr.hh"
#include "geocel/GeantGeoUtils.hh"

#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void GenericGeoTrackingResult::print_expected()
{
    using std::cout;
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
/*!
 * Get the basename or unique geometry key (defaults to suite name).
 */
auto GenericGeoTestInterface::geometry_basename() const -> std::string
{
    // Get filename based on unit test name
    ::testing::TestInfo const* const test_info
        = ::testing::UnitTest::GetInstance()->current_test_info();
    CELER_ASSERT(test_info);
    return test_info->test_case_name();
}

//---------------------------------------------------------------------------//
/*!
 * Get all logical volume labels.
 */
std::vector<std::string> GenericGeoTestInterface::get_volume_labels() const
{
    std::vector<std::string> result;

    auto const& volumes = this->geometry_interface()->volumes();
    for (auto vidx : range(this->volume_offset(), volumes.size()))
    {
        Label const& lab = volumes.at(VolumeId{vidx});
        if (!lab.empty())
        {
            result.emplace_back(to_string(lab));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all physical volume labels, including extensions.
 */
std::vector<std::string>
GenericGeoTestInterface::get_volume_instance_labels() const
{
    std::vector<std::string> result;

    auto const& vol_inst = this->geometry_interface()->volume_instances();
    for (auto vidx : range(this->volume_instance_offset(), vol_inst.size()))
    {
        Label const& lab = vol_inst.at(VolumeInstanceId{vidx});
        if (!lab.empty())
        {
            result.emplace_back(to_string(lab));
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all Geant4 PV names corresponding to volume instances.
 */
std::vector<std::string> GenericGeoTestInterface::get_g4pv_labels() const
{
    auto* world = this->g4world();
    CELER_VALIDATE(world,
                   << "cannot get g4pv names from " << this->geometry_type()
                   << " geometry: Geant4 world has not been set");

#if CELERITAS_USE_GEANT4
    auto pv_labels = make_physical_vol_labels(*world);

    auto& geo = *this->geometry_interface();
    auto const& vol_inst = geo.volume_instances();

    std::vector<std::string> result;
    for (auto vidx : range(this->volume_instance_offset(), vol_inst.size()))
    {
        if (vol_inst.at(VolumeInstanceId{vidx}).empty())
        {
            // Skip "virtual" PV
            continue;
        }

        result.push_back([&] {
            using namespace std::literals;

            G4VPhysicalVolume const* pv = geo.id_to_pv(VolumeInstanceId{vidx});
            if (!pv)
            {
                return "<null>"s;
            }
            auto id = static_cast<std::size_t>(pv->GetInstanceID());
            if (id >= pv_labels.size())
            {
                return "<out of range: "s + pv->GetName() + ">"s;
            }
            auto const& label = pv_labels[id];
            if (label.empty())
            {
                return "<not visited: "s + pv->GetName() + ">"s;
            }
            return to_string(label);
        }());
    }
    return result;
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume name, adjusting for offsets from loading multiple geo.
 */
std::string_view GenericGeoTestInterface::get_volume_name(VolumeId i) const
{
    CELER_EXPECT(i);
    auto const& volumes = this->geometry_interface()->volumes();
    auto index = this->volume_offset() + i.get();
    if (index >= volumes.size())
    {
        return "<out of range>";
    }
    return volumes.at(VolumeId{index}).name;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
