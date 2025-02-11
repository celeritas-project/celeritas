//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/VisitVolumes.test.cc
//---------------------------------------------------------------------------//
#include "geocel/g4/VisitVolumes.hh"

#include <string>
#include <vector>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/io/Label.hh"

#include "GeantGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
struct LogicalVisitor
{
    std::vector<std::string> names;
    void operator()(G4LogicalVolume const& lv)
    {
        names.push_back(lv.GetName());
    }
};

struct PhysicalVisitor
{
    std::vector<std::string> names;
    bool operator()(G4VPhysicalVolume const& pv, int depth)
    {
        std::ostringstream os;
        os << depth << ':' << pv.GetName();
        names.push_back(std::move(os).str());
        return true;
    }
};

struct MaxPhysicalVisitor : PhysicalVisitor
{
    std::unordered_map<G4VPhysicalVolume const*, int> max_depth;

    bool operator()(G4VPhysicalVolume const& pv, int depth)
    {
        auto&& [iter, inserted] = max_depth.insert({&pv, depth});
        if (!inserted)
        {
            if (iter->second >= depth)
            {
                // Already visited PV at this depth or more
                return false;
            }
            // Update the max depth
            iter->second = depth;
        }
        return PhysicalVisitor::operator()(pv, depth);
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
class VisitGeantVolumesTest : public GeantGeoTestBase
{
  public:
    using SpanStringView = Span<std::string_view const>;

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }
};

//---------------------------------------------------------------------------//
class FourLevelsTest : public VisitGeantVolumesTest
{
    std::string geometry_basename() const override { return "four-levels"; }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, logical)
{
    LogicalVisitor visit;
    visit_volumes(visit, *this->geometry()->world());

    static char const* const expected_names[]
        = {"World", "Envelope", "Shape1", "Shape2"};
    EXPECT_VEC_EQ(expected_names, visit.names);
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, physical)
{
    PhysicalVisitor visit;
    visit_volume_instances(visit, *this->geometry()->world());

    static char const* const expected_names[] = {
        "0:World_PV", "1:env1",   "2:Shape1", "3:Shape2", "1:env2",
        "2:Shape1",   "3:Shape2", "1:env3",   "2:Shape1", "3:Shape2",
        "1:env4",     "2:Shape1", "3:Shape2", "1:env5",   "2:Shape1",
        "3:Shape2",   "1:env6",   "2:Shape1", "3:Shape2", "1:env7",
        "2:Shape1",   "3:Shape2", "1:env8",   "2:Shape1", "3:Shape2",
    };
    EXPECT_VEC_EQ(expected_names, visit.names);
}

//---------------------------------------------------------------------------//
class MultiLevelTest : public VisitGeantVolumesTest
{
    std::string geometry_basename() const override { return "multi-level"; }
};

//---------------------------------------------------------------------------//

TEST_F(MultiLevelTest, logical)
{
    LogicalVisitor visit;
    visit_volumes(visit, *this->geometry()->world());

    static char const* const expected_names[] = {"world", "sph", "box"};
    EXPECT_VEC_EQ(expected_names, visit.names);
}

//---------------------------------------------------------------------------//

TEST_F(MultiLevelTest, physical)
{
    PhysicalVisitor visit;
    visit_volume_instances(visit, *this->geometry()->world());

    static char const* const expected_names[] = {
        "0:world_PV",
        "1:topsph1",
        "1:topbox1",
        "2:boxsph1",
        "2:boxsph2",
        "1:topbox2",
        "2:boxsph1",
        "2:boxsph2",
        "1:topbox3",
        "2:boxsph1",
        "2:boxsph2",
        "1:topbox4",
        "2:boxsph1",
        "2:boxsph2",
    };
    EXPECT_VEC_EQ(expected_names, visit.names);

    MaxPhysicalVisitor visit_max;
    visit_volume_instances(visit_max, *this->geometry()->world());
    static std::string const expected_max_names[] = {
        "0:world_PV",
        "1:topsph1",
        "1:topbox1",
        "2:boxsph1",
        "2:boxsph2",
        "1:topbox2",
        "1:topbox3",
        "1:topbox4",
    };
    EXPECT_VEC_EQ(expected_max_names, visit_max.names);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
