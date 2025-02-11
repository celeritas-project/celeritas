//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GenericGeoTestInterface.cc
//---------------------------------------------------------------------------//
#include "GenericGeoTestInterface.hh"

#include "corecel/io/Repr.hh"

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
 * Get all logical volume names.
 */
std::vector<std::string> GenericGeoTestInterface::get_volume_names() const
{
    std::vector<std::string> result;

    auto const& volumes = this->geometry_interface()->volumes();
    for (auto vidx : range(this->volume_offset(), volumes.size()))
    {
        result.push_back(volumes.at(VolumeId{vidx}).name);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get all physical volume names.
 */
std::vector<std::string>
GenericGeoTestInterface::get_volume_instance_names() const
{
    std::vector<std::string> result;

    auto const& vol_inst = this->geometry_interface()->volume_instances();
    for (auto vidx : range(this->volume_instance_offset(), vol_inst.size()))
    {
        result.push_back(vol_inst.at(VolumeInstanceId{vidx}).name);
    }
    return result;
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
