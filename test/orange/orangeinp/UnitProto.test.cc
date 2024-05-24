//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/UnitProto.hh"

#include <fstream>
#include <memory>
#include <sstream>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Join.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/orangeinp/CsgObject.hh"
#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Transformed.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"

#include "CsgTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
// Type aliases
//---------------------------------------------------------------------------//
using SPConstObject = std::shared_ptr<ObjectInterface const>;
using SPConstProto = std::shared_ptr<ProtoInterface const>;

//---------------------------------------------------------------------------//
// Construction helper functions
//---------------------------------------------------------------------------//
template<class CR, class... Args>
SPConstObject make_shape(std::string&& label, Args&&... args)
{
    return std::make_shared<Shape<CR>>(std::move(label),
                                       CR{std::forward<Args>(args)...});
}

SPConstObject make_translated(SPConstObject&& obj, Real3 const& trans)
{
    return std::make_shared<Transformed>(std::move(obj), Translation{trans});
}

SPConstObject make_sph(std::string&& label, real_type radius)
{
    return make_shape<Sphere>(std::move(label), radius);
}

SPConstObject
make_cyl(std::string&& label, real_type radius, real_type halfheight)
{
    return make_shape<Cylinder>(std::move(label), radius, halfheight);
}

SPConstObject make_box(std::string&& label, Real3 const& lo, Real3 const& hi)
{
    Real3 half_height = (hi - lo) / 2;
    CELER_VALIDATE(
        half_height[0] > 0 && half_height[1] > 0 && half_height[2] > 0,
        << "invalid box coordinates " << lo << ", " << hi);
    Real3 center = (hi + lo) / 2;
    auto result = make_shape<Box>(std::move(label), half_height);
    if (!soft_zero(norm(center)))
    {
        result = make_translated(std::move(result), center);
    }
    return result;
}

SPConstProto make_daughter(std::string label)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph(label + ":ext", 1);
    inp.background.fill = MaterialId{0};
    inp.label = std::move(label);

    return std::make_shared<UnitProto>(std::move(inp));
}

std::string proto_labels(ProtoInterface::VecProto const& vp)
{
    auto stream_proto_ptr = [](std::ostream& os, ProtoInterface const* p) {
        if (!p)
        {
            os << "<null>";
        }
        else
        {
            os << p->label();
        }
    };
    return to_string(join_stream(vp.begin(), vp.end(), ",", stream_proto_ptr));
}

UnitProto::MaterialInput
make_material(SPConstObject&& obj, MaterialId::size_type m)
{
    CELER_EXPECT(obj);
    UnitProto::MaterialInput result;
    result.interior = std::move(obj);
    result.fill = MaterialId{m};
    return result;
}

//---------------------------------------------------------------------------//
class UnitProtoTest : public ::celeritas::test::Test
{
  protected:
    using Unit = detail::CsgUnit;
    using Tol = Tolerance<>;

    Tolerance<> tol_ = Tol::from_relative(1e-5);
};

//---------------------------------------------------------------------------//
using LeafTest = UnitProtoTest;

TEST_F(LeafTest, errors)
{
    EXPECT_THROW(UnitProto(UnitProto::Input{}), RuntimeError);

    {
        SCOPED_TRACE("infinite global box");
        UnitProto::Input inp;
        inp.label = "leaf";
        inp.boundary.interior = std::make_shared<NegatedObject>(
            "bad-interior", make_cyl("bound", 1.0, 1.0));
        inp.boundary.zorder = ZOrder::media;
        inp.materials.push_back(
            make_material(SPConstObject(inp.boundary.interior), 1));
        UnitProto const proto{std::move(inp)};

        EXPECT_THROW(proto.build(tol_, BBox{}), RuntimeError);
    }
}

// All space is explicitly accounted for
TEST_F(LeafTest, explicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_cyl("bound", 1.0, 1.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "leaf";
    inp.materials.push_back(make_material(
        make_translated(make_cyl("bottom", 1, 0.5), {0, 0, -0.5}), 1));
    inp.materials.push_back(make_material(
        make_translated(make_cyl("top", 1, 0.5), {0, 0, 0.5}), 2));
    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, BBox{});

        static char const* const expected_surface_strings[]
            = {"Plane: z=-1", "Plane: z=1", "Cyl z: r=1", "Plane: z=0"};
        static char const* const expected_volume_strings[]
            = {"!all(+0, -1, -2)", "all(+0, -2, -3)", "all(-1, -2, +3)"};
        static char const* const expected_md_strings[] = {
            "",
            "",
            "bottom@mz,bound@mz",
            "bound@pz,top@pz",
            "",
            "bottom@cz,bound@cz,top@cz",
            "",
            "bound",
            "[EXTERIOR]",
            "bottom@pz,top@mz",
            "",
            "bottom",
            "top",
        };
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m1", "m2"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_EQ(MaterialId{}, u.background);
    }
    {
        auto u = proto.build(tol_, BBox{{-2, -2, -1}, {2, 2, 1}});
        static char const* const expected_volume_strings[] = {"F", "-3", "+3"};

        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

// Inside of the "mother" volume is implicit
TEST_F(LeafTest, implicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_cyl("bound", 1.0, 1.0);
    inp.boundary.zorder = ZOrder::exterior;
    inp.background.fill = MaterialId{0};
    inp.label = "leaf";
    inp.materials.push_back(make_material(make_cyl("middle", 1, 0.5), 1));
    UnitProto const proto{std::move(inp)};

    {
        auto u = proto.build(tol_, BBox{});

        static char const* const expected_surface_strings[] = {
            "Plane: z=-1",
            "Plane: z=1",
            "Cyl z: r=1",
            "Plane: z=-0.5",
            "Plane: z=0.5",
        };
        static char const* const expected_volume_strings[]
            = {"!all(+0, -1, -2)", "all(-2, +3, -4)"};
        static char const* const expected_fill_strings[]
            = {"<UNASSIGNED>", "m1"};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_EQ(MaterialId{0}, u.background);
    }
    {
        auto u = proto.build(tol_, BBox{{-2, -2, -1}, {2, 2, 1}});

        static char const* const expected_volume_strings[]
            = {"F", "all(+3, -4)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_EQ(MaterialId{0}, u.background);
    }
}

//---------------------------------------------------------------------------//
using MotherTest = UnitProtoTest;

TEST_F(MotherTest, explicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "mother";
    inp.materials.push_back(
        make_material(make_translated(make_sph("leaf", 1), {0, 0, -5}), 1));
    inp.materials.push_back(
        make_material(make_translated(make_sph("leaf2", 1), {0, 0, 5}), 2));
    inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
    inp.daughters.push_back(
        {make_daughter("d2"),
         Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});

    // Construct "inside" cell
    std::vector<std::pair<Sense, SPConstObject>> interior
        = {{Sense::inside, inp.boundary.interior}};
    for (auto const& m : inp.materials)
    {
        interior.push_back({Sense::outside, m.interior});
    }
    for (auto const& d : inp.daughters)
    {
        interior.push_back({Sense::outside, d.make_interior()});
    }
    inp.materials.push_back(
        make_material(make_rdv("interior", std::move(interior)), 3));

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1,d2", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, BBox{});

        static char const* const expected_surface_strings[] = {
            "Sphere: r=10",
            "Sphere: r=1 at {0,5,0}",
            "Sphere: r=1 at {0,-5,0}",
            "Sphere: r=1 at {0,0,-5}",
            "Sphere: r=1 at {0,0,5}",
        };
        static char const* const expected_volume_strings[] = {
            "+0",
            "-1",
            "-2",
            "-3",
            "-4",
            "all(-0, +1, +2, +3, +4)",
        };
        static char const* const expected_md_strings[] = {"",
                                                          "",
                                                          "[EXTERIOR],bound@s",
                                                          "bound",
                                                          "d1:ext@s",
                                                          "d1:ext",
                                                          "d2:ext@s",
                                                          "d2:ext",
                                                          "leaf@s",
                                                          "leaf",
                                                          "leaf2@s",
                                                          "leaf2",
                                                          "interior"};
        static char const* const expected_trans_strings[] = {
            "2: t=0 -> {}",
            "3: t=0",
            "4: t=0",
            "5: t=1 -> {{0,5,0}}",
            "6: t=0",
            "7: t=2 -> {{{1,0,0},{0,0,-1},{0,1,0}}, {0,-5,0}}",
            "8: t=0",
            "9: t=3 -> {{0,0,-5}}",
            "10: t=0",
            "11: t=4 -> {{0,0,5}}",
            "12: t=0",
        };
        static char const* const expected_fill_strings[] = {
            "<UNASSIGNED>",
            "{u=0, t=1}",
            "{u=1, t=2}",
            "m1",
            "m2",
            "m3",
        };
        static int const expected_volume_nodes[] = {2, 5, 7, 9, 11, 12};

        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
        EXPECT_VEC_EQ(expected_trans_strings, transform_strings(u));
        EXPECT_VEC_EQ(expected_fill_strings, fill_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(MaterialId{}, u.background);
    }
    {
        auto u = proto.build(tol_, BBox{{-10, -10, -10}, {10, 10, 10}});
        static char const* const expected_volume_strings[]
            = {"F", "-1", "-2", "-3", "-4", "all(+1, +2, +3, +4)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

TEST_F(MotherTest, implicit_exterior)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "mother";
    inp.materials.push_back(
        make_material(make_translated(make_sph("leaf", 1), {0, 0, -5}), 1));
    inp.materials.push_back(
        make_material(make_translated(make_sph("leaf2", 1), {0, 0, 5}), 2));
    inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
    inp.daughters.push_back(
        {make_daughter("d2"),
         Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});
    inp.background.fill = MaterialId{3};

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1,d2", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, BBox{});
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "-2", "-3", "-4"};
        static int const expected_volume_nodes[] = {2, 5, 7, 9, 11};

        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_volume_nodes, volume_nodes(u));
        EXPECT_EQ(MaterialId{3}, u.background);
    }
    {
        auto u = proto.build(tol_, BBox{{-10, -10, -10}, {10, 10, 10}});
        static char const* const expected_volume_strings[]
            = {"F", "-1", "-2", "-3", "-4"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

TEST_F(MotherTest, fuzziness)
{
    UnitProto::Input inp;
    inp.boundary.interior = make_sph("bound", 10.0);
    inp.boundary.zorder = ZOrder::media;
    inp.label = "fuzzy";
    inp.daughters.push_back({make_daughter("d1"), {}});
    inp.materials.push_back(make_material(
        make_rdv("interior",
                 {{Sense::inside, inp.boundary.interior},
                  {Sense::outside, make_sph("similar", 1.0001)}}),
        1));

    UnitProto const proto{std::move(inp)};

    EXPECT_EQ("d1", proto_labels(proto.daughters()));

    {
        auto u = proto.build(tol_, BBox{});
        static char const* const expected_surface_strings[]
            = {"Sphere: r=10", "Sphere: r=1", "Sphere: r=1.0001"};
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "all(-0, +2)"};
        static char const* const expected_md_strings[] = {"",
                                                          "",
                                                          "[EXTERIOR],bound@s",
                                                          "bound",
                                                          "d1:ext@s",
                                                          "d1:ext",
                                                          "similar@s",
                                                          "similar",
                                                          "interior"};
        EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
        EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    }
    {
        // Simplify with lower tolerance because the user has tried to avoid
        // overlap by adding .0001 to the "similar" shape
        auto u = proto.build(Tol::from_relative(1e-3), BBox{});
        static char const* const expected_volume_strings[]
            = {"+0", "-1", "all(-0, +1)"};
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }
}

//---------------------------------------------------------------------------//
class InputBuilderTest : public UnitProtoTest
{
  public:
    void run_test(UnitProto const& global)
    {
        OrangeInput inp = build_input(tol_, global);
        EXPECT_TRUE(inp);
        if (!CELERITAS_USE_JSON)
        {
            GTEST_SKIP() << "JSON is disabled: cannot compare output";
        }

        std::string const ref_path = this->test_data_path("orange", "")
                                     + this->make_unique_filename(".org.json");

        // Export input to JSON
        std::ostringstream actual;
        actual << inp;

        // Read 'gold' file
        std::ifstream ref{ref_path};
        if (!ref)
        {
            ADD_FAILURE() << "failed to open reference file at '" << ref_path
                          << "': writing gold file instead";
            ref.close();

            std::ofstream out(ref_path);
            CELER_VALIDATE(out,
                           << "failed to open gold file '" << ref_path
                           << "' for writing");
            out << actual.str();
            return;
        }

        std::stringstream expected;
        expected << ref.rdbuf();
        EXPECT_JSON_EQ(expected.str(), actual.str())
            << "update the file at " << ref_path;
    }
};

TEST_F(InputBuilderTest, globalspheres)
{
    UnitProto global{[] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 10.0);
        inp.boundary.zorder = ZOrder::media;
        inp.label = "global";

        auto inner = make_sph("inner", 5.0);

        // Construct "inside" cell
        inp.materials.push_back(
            make_material(make_rdv("shell",
                                   {{Sense::inside, inp.boundary.interior},
                                    {Sense::outside, inner}}),
                          1));
        inp.materials.push_back(make_material(std::move(inner), 2));
        return inp;
    }()};

    this->run_test(global);
}

TEST_F(InputBuilderTest, bgspheres)
{
    UnitProto global{[] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 10.0);
        inp.label = "global";

        inp.materials.push_back(make_material(
            make_translated(make_sph("top", 2.0), {0, 0, 3}), 1));
        inp.materials.push_back(make_material(
            make_translated(make_sph("bottom", 3.0), {0, 0, -3}), 2));
        inp.background.fill = MaterialId{3};
        return inp;
    }()};

    this->run_test(global);
}

// Equivalent to universes.org.omn
TEST_F(InputBuilderTest, universes)
{
    auto most_inner = std::make_shared<UnitProto>([] {
        auto patricia = make_box("patricia", {0.0, 0.0, 0.0}, {0.5, 0.5, 1});

        UnitProto::Input inp;
        inp.label = "most_inner";
        inp.boundary.interior = patricia;
        inp.boundary.zorder = ZOrder::media;
        inp.materials.push_back(
            make_material(make_rdv("patty", {{Sense::inside, patricia}}), 2));
        return inp;
    }());

    auto inner = std::make_shared<UnitProto>([&] {
        auto alpha = make_box("alpha", {-1, -1, 0}, {1, 1, 1});
        auto beta = make_box("beta", {1, -1, 0}, {3, 1, 1});
        auto gamma = make_box("gamma", {-2, -2, 0}, {4, 2, 1});

        UnitProto::Input inp;
        inp.label = "inner";
        inp.boundary.interior = gamma;
        inp.boundary.zorder = ZOrder::media;
        inp.daughters.push_back({most_inner, Translation{{-2, -2, 0}}});
        inp.materials.push_back(
            make_material(make_rdv("a", {{Sense::inside, alpha}}), 0));
        inp.materials.push_back(
            make_material(make_rdv("b", {{Sense::inside, beta}}), 1));
        inp.materials.push_back(make_material(
            make_rdv("c",
                     {{Sense::outside, alpha},
                      {Sense::outside, beta},
                      {Sense::inside, gamma},
                      {Sense::outside, inp.daughters[0].make_interior()}}),
            2));
        return inp;
    }());

    auto outer = std::make_shared<UnitProto>([&] {
        auto bob = make_box("bob", {0, 0, -0.5}, {6, 2, 1.5});
        auto john = make_box("john", {-2, -6, -1}, {8, 4, 2});

        UnitProto::Input inp;
        inp.label = "outer";
        inp.boundary.interior = john;
        inp.boundary.zorder = ZOrder::media;
        inp.daughters.push_back(
            {inner, Translation{{2, -2, -0.5}}, ZOrder::media});
        inp.daughters.push_back(
            {inner, Translation{{2, -2, 0.5}}, ZOrder::media});
        inp.materials.push_back(
            make_material(make_rdv("bobby", {{Sense::inside, bob}}), 3));
        inp.materials.push_back(make_material(
            make_rdv("johnny",
                     {{Sense::outside, bob},
                      {Sense::inside, john},
                      {Sense::outside, inp.daughters[0].make_interior()},
                      {Sense::outside, inp.daughters[1].make_interior()}}),
            4));
        return inp;
    }());

    this->run_test(*outer);
}

TEST_F(InputBuilderTest, hierarchy)
{
    auto leaf = std::make_shared<UnitProto>([] {
        UnitProto::Input inp;
        inp.boundary.interior = make_cyl("bound", 1.0, 1.0);
        inp.boundary.zorder = ZOrder::media;
        inp.label = "leafy";
        inp.materials.push_back(make_material(
            make_translated(make_cyl("bottom", 1, 0.5), {0, 0, -0.5}), 1));
        inp.materials.push_back(make_material(
            make_translated(make_cyl("top", 1, 0.5), {0, 0, 0.5}), 2));
        return inp;
    }());

    auto filled_daughter = std::make_shared<UnitProto>([] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 10.0);
        inp.boundary.zorder = ZOrder::exterior;
        inp.label = "filled_daughter";
        inp.materials.push_back(make_material(
            make_translated(make_sph("leaf1", 1), {0, 0, -5}), 1));
        inp.materials.push_back(make_material(
            make_translated(make_sph("leaf2", 1), {0, 0, 5}), 2));
        inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
        inp.daughters.push_back(
            {make_daughter("d2"),
             Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});
        inp.background.fill = MaterialId{3};
        return inp;
    }());

    auto global = std::make_shared<UnitProto>([&] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 100.0);
        inp.boundary.zorder = ZOrder::media;
        inp.label = "global";
        inp.daughters.push_back({make_daughter("d1"), Translation{{0, 5, 0}}});
        inp.daughters.push_back(
            {make_daughter("d2"),
             Transformation{make_rotation(Axis::x, Turn{0.25}), {0, -5, 0}}});
        inp.daughters.push_back({filled_daughter, Translation{{0, 0, -20}}});
        inp.daughters.push_back({leaf, Translation{{0, 0, 20}}});

        inp.materials.push_back(make_material(
            make_translated(make_sph("leaf1", 1), {0, 0, -5}), 1));

        // Construct "inside" cell
        inp.materials.push_back(make_material(
            make_rdv("interior",
                     [&] {
                         VecSenseObj interior
                             = {{Sense::inside, inp.boundary.interior}};
                         for (auto const& d : inp.daughters)
                         {
                             interior.push_back(
                                 {Sense::outside, d.make_interior()});
                         }
                         for (auto const& m : inp.materials)
                         {
                             interior.push_back({Sense::outside, m.interior});
                         }
                         return interior;
                     }()),
            3));

        return inp;
    }());

    this->run_test(*global);
}

// Equivalent to universes.org.omn
TEST_F(InputBuilderTest, incomplete_bb)
{
    auto inner = std::make_shared<UnitProto>([] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 5.0);
        inp.boundary.zorder = ZOrder::media;
        inp.label = "inner";

        using VR2 = GenTrap::VecReal2;
        auto trd = make_shape<GenTrap>("turd",
                                       real_type{3},
                                       VR2{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}},
                                       VR2{{-2, -2}, {2, -2}, {2, 2}, {-2, 2}});
        inp.materials.push_back(
            make_material(make_rdv("fill",
                                   {{Sense::inside, inp.boundary.interior},
                                    {Sense::outside, trd}}),
                          1));
        inp.materials.push_back(make_material(std::move(trd), 2));
        return inp;
    }());

    auto outer = std::make_shared<UnitProto>([&] {
        UnitProto::Input inp;
        inp.boundary.interior = make_sph("bound", 10.0);
        inp.boundary.zorder = ZOrder::media;
        inp.label = "global";

        inp.daughters.push_back({inner, Translation{{2, 0, 0}}});

        inp.materials.push_back(make_material(
            make_rdv("shell",
                     {{Sense::inside, inp.boundary.interior},
                      {Sense::outside, inp.daughters.front().make_interior()}}),
            1));
        return inp;
    }());

    this->run_test(*outer);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
