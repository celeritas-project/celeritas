//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/DiagnosticGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Count invocations to find intersections and safety distances.
 */
template<class GTV>
class DiagnosticGeoTrackView : public GTV
{
  public:
    using DetailedInitializer = typename GTV::DetailedInitializer;

  public:
    //! Forward construction arguments to the original stepper
    template<class... Args>
    DiagnosticGeoTrackView(Args&&... args) : GTV(std::forward<Args>(args)...)
    {
    }

    // Initialize the state
    inline CELER_FUNCTION DiagnosticGeoTrackView&
    operator=(GeoTrackInitializer const& init)
    {
        GTV::operator=(init);
        return *this;
    }
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION DiagnosticGeoTrackView&
    operator=(DetailedInitializer const& init)
    {
        GTV::operator=(init);
        return *this;
    }

    //!@{
    //! Forward to base class and increment counter
    Propagation find_next_step()
    {
        ++num_intersect_;
        return GTV::find_next_step();
    }

    Propagation find_next_step(real_type max_step)
    {
        ++num_intersect_;
        return GTV::find_next_step(max_step);
    }

    real_type find_safety()
    {
        ++num_safety_;
        return GTV::find_safety();
    }

    real_type find_safety(real_type max_step)
    {
        ++num_safety_;
        return GTV::find_safety(max_step);
    }
    //!@}

    //! Number of calls fo find_next_step
    size_type intersect_count() const { return num_intersect_; }
    //! Number of calls fo find_safety
    size_type safety_count() const { return num_safety_; }
    //! Reset the stepscounter
    void reset_count() { num_intersect_ = num_safety_ = 0; }

  private:
    size_type num_intersect_{0};
    size_type num_safety_{0};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class GTV>
CELER_FUNCTION DiagnosticGeoTrackView(GTV&&)->DiagnosticGeoTrackView<GTV>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
