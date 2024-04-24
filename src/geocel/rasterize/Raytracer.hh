//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

    size_type pixel_;

    //// HELPER FUNCTIONS ////

    // Initialize the geometry at this pixel index
    inline CELER_FUNCTION void initialize_at_pixel(size_type pix);

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
    , pixel_{invalid_pixel()}
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
    }

    real_type pix_dist = image_.pixel_width();
    real_type max_dist = 0;
    int cur_id = this->calc_id_(geo_);
    int max_id = cur_id;
    auto abort_counter = this->max_crossings_per_pixel();

    auto next = geo_.find_next_step(pix_dist);
    while (next.boundary && pix_dist > 0)
    {
        CELER_ASSERT(next.distance <= pix_dist);
        // Move to geometry boundary
        pix_dist -= next.distance;

        if (max_id == cur_id)
        {
            max_dist += next.distance;
        }
        else if (next.distance > max_dist)
        {
            max_dist = next.distance;
            max_id = cur_id;
        }

        // Cross surface and update post-crossing ID
        geo_.move_to_boundary();
        geo_.cross_boundary();
        cur_id = this->calc_id_(geo_);

        if (--abort_counter == 0)
        {
            // Give up and move to end of pixel, reinitialize on the next call
            pix_dist = 0;
        }
        if (pix_dist > 0)
        {
            // Next movement is to end of geo_ or pixel
            next = geo_.find_next_step(pix_dist);
        }
    }

    if (pix_dist > 0)
    {
        // Move to pixel boundary
        geo_.move_internal(pix_dist);
        if (pix_dist > max_dist)
        {
            max_dist = pix_dist;
            max_id = cur_id;
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
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
