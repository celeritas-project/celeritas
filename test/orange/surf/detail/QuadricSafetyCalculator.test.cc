//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadricSafetyCalculator.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/QuadricSafetyCalculator.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/GeneralQuadric.hh"
#include "orange/surf/SimpleQuadric.hh"
#include "orange/surf/Sphere.hh"

#include "celeritas_test.hh"
#include "../SurfaceTestUtils.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
//! Safety and reference distance sampled along a line through a surface
struct LineData
{
    std::string surface;
    std::string reference;  //!< "exact" or "sampled"
    Real3 start;
    Real3 dir;
    std::vector<real_type> t;
    std::vector<real_type> safety;
    std::vector<real_type> distance;
    std::vector<int> inside;  //!< Whether the point has an "inside" sense
};

//---------------------------------------------------------------------------//
void write_array(std::ostream& os, Real3 const& v)
{
    os << '[' << v[0] << ", " << v[1] << ", " << v[2] << ']';
}

template<class T>
void write_array(std::ostream& os, std::vector<T> const& v)
{
    os << '[';
    for (auto i : range(v.size()))
    {
        os << (i == 0 ? "" : ", ") << v[i];
    }
    os << ']';
}

//! Write line data as JSON for plotting
void write_json(std::ostream& os, std::vector<LineData> const& lines)
{
    os << std::setprecision(10) << "[\n";
    for (auto i : range(lines.size()))
    {
        auto const& line = lines[i];
        os << R"( {"surface": ")" << line.surface << R"(", "reference": ")"
           << line.reference << R"(",)" << "\n  \"start\": ";
        write_array(os, line.start);
        os << ", \"dir\": ";
        write_array(os, line.dir);
        os << ",\n  \"t\": ";
        write_array(os, line.t);
        os << ",\n  \"safety\": ";
        write_array(os, line.safety);
        os << ",\n  \"distance\": ";
        write_array(os, line.distance);
        os << ",\n  \"inside\": ";
        write_array(os, line.inside);
        os << '}' << (i + 1 < lines.size() ? "," : "") << '\n';
    }
    os << "]\n";
}

//---------------------------------------------------------------------------//
class QuadricSafetyCalculatorTest : public ::celeritas::test::Test
{
  protected:
    static constexpr int num_points = 51;

    /*!
     * Sample the safety and a reference distance along a line segment.
     *
     * The reference is either the exact distance or a sampled distance (which
     * is never smaller than the exact distance); the safety must not exceed
     * it. Where \c is_exact returns true for a point, the safety must equal
     * the exact distance.
     */
    template<class S, class F, class E>
    void sample_line(char const* surface,
                     char const* reference,
                     S const& surf,
                     F&& calc_distance,
                     E&& is_exact,
                     Real3 const& start,
                     Real3 const& dir,
                     real_type length)
    {
        SCOPED_TRACE(surface);
        LineData line{surface, reference, start, make_unit_vector(dir)};
        SoftEqual<> soft_eq;
        for (auto i : range(num_points))
        {
            real_type const t = length * i / (num_points - 1);
            Real3 const pos = start + t * line.dir;
            real_type const safety = surf.calc_safety(pos);
            real_type const distance = calc_distance(pos);

            EXPECT_GE(safety, 0) << "at " << repr(pos);
            EXPECT_TRUE(safety <= distance || soft_eq(distance, safety))
                << "safety " << safety << " exceeds " << reference
                << " distance " << distance << " at " << repr(pos);
            if (is_exact(pos))
            {
                EXPECT_SOFT_EQ(distance, safety) << "at " << repr(pos);
            }

            line.t.push_back(t);
            line.safety.push_back(safety);
            line.distance.push_back(distance);
            line.inside.push_back(surf.calc_sense(pos) == SignedSense::inside);
        }
        lines_.push_back(std::move(line));
    }

    //! Sample the distance by firing rays in many directions
    template<class S>
    static auto sampled(S const& surf)
    {
        return [&surf](Real3 const& pos) {
            return ::celeritas::test::calc_sampled_distance(surf, pos);
        };
    }

    std::vector<LineData> lines_;
};

//---------------------------------------------------------------------------//
TEST_F(QuadricSafetyCalculatorTest, eigen_bounds)
{
    Real3 const grad{3, 4, 0};

    // Planar (no curvature): distance is |f|/|grad f|
    QuadricSafetyCalculator calc_plane{0, 0};
    EXPECT_SOFT_EQ(0.2, calc_plane(1, grad));
    EXPECT_SOFT_EQ(0.2, calc_plane(-1, grad));
    EXPECT_SOFT_EQ(0, calc_plane(0, grad));

    // Positive curvature only reduces the safety on the negative side
    QuadricSafetyCalculator calc_convex{1, 1};
    EXPECT_SOFT_EQ(0.2, calc_convex(1, grad));
    EXPECT_SOFT_EQ(2 / (5 + std::sqrt(real_type(29))), calc_convex(-1, grad));

    // Negative curvature only reduces the safety on the positive side
    QuadricSafetyCalculator calc_concave{-1, -1};
    EXPECT_SOFT_EQ(2 / (5 + std::sqrt(real_type(29))), calc_concave(1, grad));
    EXPECT_SOFT_EQ(0.2, calc_concave(-1, grad));

    // At a critical point the curvature alone bounds the distance
    QuadricSafetyCalculator calc_saddle{-4, 1};
    EXPECT_SOFT_EQ(0.5, calc_saddle(1, {0, 0, 0}));
    EXPECT_SOFT_EQ(1, calc_saddle(-1, {0, 0, 0}));
}

//---------------------------------------------------------------------------//
/*!
 * Compare safety with the exact or sampled distance along lines.
 *
 * The output JSON file is a list of lines, each with a surface label,
 * reference type, start point, direction, and arrays of the distance along the
 * line, the safety, and the reference distance.
 */
TEST_F(QuadricSafetyCalculatorTest, along_lines)
{
    auto never = [](Real3 const&) { return false; };
    {
        // Sphere of radius 2: exact inside
        Sphere const sph{{0, 0, 0}, 2};
        SimpleQuadric const sq{sph};
        auto inside = [&sph](Real3 const& pos) {
            return sph.calc_sense(pos) == SignedSense::inside;
        };
        this->sample_line(
            "sphere",
            "exact",
            sq,
            [&sph](Real3 const& pos) { return sph.calc_safety(pos); },
            inside,
            {-4, 0.5, 0},
            {1, 0, 0},
            8);
    }
    {
        // Cone along z with a half-angle of atan(0.5)
        ConeZ const cone{{0, 0, 0}, 0.5};
        SimpleQuadric const sq{cone};
        auto exact
            = [&cone](Real3 const& pos) { return cone.calc_safety(pos); };
        this->sample_line("cone (across)",
                          "exact",
                          sq,
                          exact,
                          never,
                          {-3, 0.2, 2},
                          {1, 0, 0},
                          6);
        this->sample_line("cone (through apex)",
                          "exact",
                          sq,
                          exact,
                          never,
                          {-2, 0, -1},
                          {2, 0, 1},
                          2 * std::sqrt(real_type(5)));
    }
    {
        // Unit cylinder along (1, 1, 0) / sqrt(2): exact inside
        GeneralQuadric const gq{{0.5, 0.5, 1}, {-1, 0, 0}, {0, 0, 0}, -1};
        auto radius = [](Real3 const& pos) {
            return std::sqrt(
                dot_product(pos, pos) - ipow<2>(pos[0] + pos[1]) / 2);
        };
        this->sample_line(
            "rotated cylinder",
            "exact",
            gq,
            [&radius](Real3 const& pos) { return std::fabs(radius(pos) - 1); },
            [&radius](Real3 const& pos) { return radius(pos) < 1; },
            {-2.5, 2.5, 0.3},
            {1, -1, 0},
            5 * std::sqrt(real_type(2)));
    }
    {
        // Ellipsoid with radii 3, 2, 1
        SimpleQuadric const sq{{4, 9, 36}, {0, 0, 0}, -36};
        this->sample_line("ellipsoid",
                          "sampled",
                          sq,
                          sampled(sq),
                          never,
                          {-5, 0.5, 0.2},
                          {1, 0, 0},
                          10);
    }
    {
        // One-sheet hyperboloid x^2 + y^2 - z^2 = 1
        SimpleQuadric const sq{{1, 1, -1}, {0, 0, 0}, -1};
        this->sample_line("hyperboloid",
                          "sampled",
                          sq,
                          sampled(sq),
                          never,
                          {-4, 0.3, 1},
                          {1, 0, 0},
                          8);
    }
    {
        // Twisted surface like the faces of an accordion calorimeter
        GeneralQuadric const gq{{0, 0, 0.1}, {0, 0.3, -0.2}, {1, 0.5, 0}, -2};
        this->sample_line("twisted",
                          "sampled",
                          gq,
                          sampled(gq),
                          never,
                          {-1, 0, 0.5},
                          {1, 0, 0},
                          6);
    }

    auto const filename = this->make_unique_filename(".json");
    std::ofstream out(filename);
    write_json(out, lines_);
    std::cout << "Wrote safety along lines to " << filename << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
