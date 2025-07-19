//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "geocel/detail/LengthUnits.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Check validity of safety and volume crossings while navigating on CPU.
 *
 * Also count the number of calls to "find distance" and "find safety".
 *
 * \note This class is instantiated in GenericXTestBase.cc for geometry type X.
 * The member function definitions are in CheckedGeoTrackView.t.hh.
 */
template<class GTV>
class CheckedGeoTrackView : public GTV
{
  public:
    //!@{
    //! \name Type aliases
    using GeoTrackViewT = GTV;
    using Initializer_t = typename GTV::Initializer_t;
    //!@}

  public:
    //! Forward construction arguments to the original track view
    template<class... Args>
    CheckedGeoTrackView(Args&&... args) : GTV(std::forward<Args>(args)...)
    {
    }

    // Initialize the state
    CheckedGeoTrackView& operator=(GeoTrackInitializer const& init);

    //! Check volume consistency this far from the boundary
    static constexpr real_type safety_tol()
    {
        return 1e-4 * lengthunits::centimeter;
    }

    // Calculate or return the safety up to an infinite distance
    real_type find_safety();

    // Calculate or return the safety up to the given distance
    real_type find_safety(real_type max_safety);

    // Change the direction
    void set_dir(Real3 const&);

    // Find the distance to the next boundary
    Propagation find_next_step();

    // Find the distance to the next boundary
    Propagation find_next_step(real_type max_distance);

    // Move a linear step fraction
    void move_internal(real_type);

    // Move within the safety distance to a specific point
    void move_internal(Real3 const& pos);

    // Move to the boundary in preparation for crossing it
    void move_to_boundary();

    // Cross from one side of the current surface to the other
    void cross_boundary();

    //! Number of calls of find_next_step
    size_type intersect_count() const { return num_intersect_; }
    //! Number of calls of find_safety
    size_type safety_count() const { return num_safety_; }
    //! Reset the stepscounter
    void reset_count() { num_intersect_ = num_safety_ = 0; }

  private:
    bool checked_internal_{false};
    size_type num_intersect_{0};
    size_type num_safety_{0};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV>
CheckedGeoTrackView(GTV&&) -> CheckedGeoTrackView<GTV>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
