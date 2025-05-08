//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Raytracer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"

#include "ImageLineView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Trace each pixel along a line.
 *
 * \tparam GTV Geometry Track View
 * \tparam F   Calculate the pixel value (result_type) given GTV
 */
template<class GTV, class F>
class Raytracer
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = int;
    //!@}

  public:
    // Construct from geo track view, function to calculate ID, and image
    inline CELER_FUNCTION
    Raytracer(GTV&& geo, F&& calc_id, ImageLineView const& image);

    // Calculate the value for the next pixel
    inline CELER_FUNCTION result_type operator()(size_type pix);

  private:
    //// DATA ////

    GTV geo_;
    F calc_id_;
    ImageLineView const& image_;

    size_type pixel_{invalid_pixel()};  //!< Current pixel
    real_type distance_{0};  //!< Distance to next boundary
    int cur_id_{-1};  //!< Current ID

    //// HELPER FUNCTIONS ////

    // Initialize the geometry at this pixel index
    inline CELER_FUNCTION void initialize_at_pixel(size_type pix);

    // Find the next step
    inline CELER_FUNCTION void find_next_step();

    //! Sentinel value for needing the pixel to be reset
    static CELER_CONSTEXPR_FUNCTION size_type invalid_pixel()
    {
        return static_cast<size_type>(-1);
    }

    //! Stop tracking the pixel if this many crossings occur
    static CELER_CONSTEXPR_FUNCTION short int max_crossings_per_pixel()
    {
        return 32;
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV, class F>
CELER_FUNCTION Raytracer(GTV&&, F&&, ImageLineView const&)
    -> Raytracer<GTV, F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from geo track view, function to calculate ID, and image.
 */
template<class GTV, class F>
CELER_FUNCTION
Raytracer<GTV, F>::Raytracer(GTV&& geo, F&& calc_id, ImageLineView const& image)
    : geo_{celeritas::forward<GTV>(geo)}
    , calc_id_{celeritas::forward<F>(calc_id)}
    , image_{image}
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the value for a single pixel.
 */
template<class GTV, class F>
CELER_FUNCTION auto Raytracer<GTV, F>::operator()(size_type pix) -> result_type
{
    if (pix != pixel_)
    {
        this->initialize_at_pixel(pix);
        if (pixel_ == invalid_pixel())
        {
            // Pixel starts outside the geometry
            return cur_id_;
        }
    }

    // Remaining distance to the end of the pixel
    real_type pix_dist = image_.pixel_width();
    // Distance associated with the largest segment so far
    real_type max_dist = 0;
    int max_id = cur_id_;
    // Countdown to limit stuck tracks or overdetailed pixels
    auto abort_counter = this->max_crossings_per_pixel();

    while (cur_id_ >= 0 && distance_ < pix_dist)
    {
        // Move to geometry boundary
        if (max_id == cur_id_)
        {
            max_dist += distance_;
        }
        else if (distance_ > max_dist)
        {
            max_dist = distance_;
            max_id = cur_id_;
        }

        // Update pixel and boundary distance
        pix_dist -= distance_;
        distance_ = 0;

        // Cross surface and update post-crossing ID
        geo_.move_to_boundary();
        geo_.cross_boundary();

        if (--abort_counter == 0)
        {
            // Give up and move to end of pixel, reinitialize on the next call
            pix_dist = 0;
        }
        if (pix_dist > 0)
        {
            // Update next distance and current ID for the remaining pixel
            this->find_next_step();
        }
    }

    if (pix_dist > 0)
    {
        // Move to pixel boundary
        distance_ -= pix_dist;
        if (pix_dist > max_dist)
        {
            max_dist = pix_dist;
            max_id = cur_id_;
        }
    }

    if (abort_counter != 0)
    {
        ++pixel_;
    }
    else
    {
        pixel_ = invalid_pixel();
    }

    return max_id;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the geometry at the given pixel position.
 */
template<class GTV, class F>
CELER_FUNCTION void Raytracer<GTV, F>::initialize_at_pixel(size_type pix)
{
    CELER_EXPECT(pix < image_.max_index());

    GeoTrackInitializer init{image_.start_pos(), image_.start_dir()};
    axpy(pix * image_.pixel_width(), init.dir, &init.pos);
    geo_ = init;
    pixel_ = pix;
    distance_ = 0;
    this->find_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next step.
 */
template<class GTV, class F>
CELER_FUNCTION void Raytracer<GTV, F>::find_next_step()
{
    CELER_EXPECT(pixel_ != invalid_pixel());
    CELER_EXPECT(distance_ <= 0);
    if (geo_.is_outside())
    {
        // Skip this pixel since not all navigation engines can trace from
        // outside the geometry
        pixel_ = invalid_pixel();
        distance_ = numeric_limits<real_type>::infinity();
    }
    else
    {
        // Search to distances just past the end of the saved pixels (so the
        // last point isn't coincident with a boundary)
        distance_ = geo_.find_next_step((image_.max_index() + 1 - pixel_)
                                        * image_.pixel_width())
                        .distance;
    }
    cur_id_ = this->calc_id_(geo_);

    CELER_ENSURE(distance_ > 0);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
