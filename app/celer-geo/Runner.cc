//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <fstream>

#include "corecel/Config.hh"

#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stopwatch.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/rasterize/RaytraceImager.hh"
#include "orange/OrangeParams.hh"
#if CELERITAS_USE_VECGEOM
#    include "geocel/vg/VecgeomParams.hh"
#endif

#define CASE_RETURN_FUNC_T(T, FUNC, ...) \
    case T:                              \
        return this->FUNC<T>(__VA_ARGS__)

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with model setup.
 */
Runner::Runner(ModelSetup const& input)
    : input_{input}, tracing_{input_.perfetto_file}
{
    // Initialize GPU
    activate_device();
    if (input_.cuda_heap_size)
    {
        set_cuda_heap_size(input_.cuda_heap_size);
    }
    if (input_.cuda_stack_size)
    {
        set_cuda_stack_size(input_.cuda_stack_size);
    }

    if (CELERITAS_USE_GEANT4 && ends_with(input_.geometry_file, ".gdml"))
    {
        // Retain the Geant4 world for possible reuse across geometries
        CELER_EXPECT(celeritas::global_geant_geo().expired());
        this->load_geometry<Geometry::geant4>();
        CELER_EXPECT(!celeritas::global_geant_geo().expired());
    }
    else
    {
        CELER_VALIDATE(std::ifstream{input_.geometry_file}.is_open(),
                       << "input model filename '" << input_.geometry_file
                       << "' does not exist");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Perform a raytrace.
 */
auto Runner::trace(TraceSetup const& setup, ImageInput const& image_inp)
    -> SPImage
{
    // Create image params
    last_image_ = std::make_shared<ImageParams>(image_inp);

    return this->trace(setup);
}

//---------------------------------------------------------------------------//
/*!
 * Perform a raytrace using the last image but a new geometry/memspace.
 *
 * The memory space in Celeritas is the same as the execution space.
 */
auto Runner::trace(TraceSetup const& setup) -> SPImage
{
    CELER_VALIDATE(last_image_,
                   << "first trace input did not specify an image");

    // Load geometry
    SPImager imager = this->make_imager(setup.geometry);

    // Create image
    SPImage image = this->make_traced_image(setup.memspace, *imager);
    return image;
}
//---------------------------------------------------------------------------//
/*!
 * Get volume names from an already loaded geometry.
 */
std::vector<std::string> Runner::get_volumes(Geometry g) const&
{
    CELER_EXPECT(geo_cache_[g]);

    auto const& geo = *geo_cache_[g];
    std::vector<std::string> result(geo.impl_volumes().size());
    for (auto i : range<ImplVolumeId::size_type>(result.size()))
    {
        result[i] = geo.impl_volumes().at(ImplVolumeId{i}).name;
    }
    return result;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Load a geometry, caching it.
 *
 * If Geant4 is available and the input file is GDML, this will be executed
 * when the runner is constructed to save a reusable pointer to the Geant4
 * world. Otherwise, this is called by the imager when raytracing a new
 * geometry type.
 */
template<Geometry G>
auto Runner::load_geometry() -> std::shared_ptr<GeoParams_t<G> const>
{
    using GP = GeoParams_t<G>;

    auto& cached = geo_cache_[G];
    std::shared_ptr<GP const> geo;
    if constexpr (!is_geometry_configured_v<GP>)
    {
        CELER_NOT_CONFIGURED(to_cstring(G));
    }
    else if (cached)
    {
        // Downcast to type
        geo = std::dynamic_pointer_cast<GP const>(cached);
    }
    else
    {
        Stopwatch get_time;
        if constexpr (G != Geometry::geant4)
        {
            if (geo_cache_[Geometry::geant4])
            {
                // Load from existing Geant4 geometry
                auto geant_geo
                    = std::dynamic_pointer_cast<GeantGeoParams const>(
                        geo_cache_[Geometry::geant4]);
                CELER_ASSUME(geant_geo);
                geo = GP::from_geant(geant_geo);
            }
        }
        if (!geo)
        {
            // Load directly from input file
            geo = GP::from_gdml(input_.geometry_file);
        }
        // Save load time
        timers_[std::string{"load_"} + to_cstring(G)] = get_time();

        // Save geometry in cache
        cached = geo;
    }

    CELER_ENSURE(cached);
    CELER_ENSURE(geo);
    return geo;
}

//---------------------------------------------------------------------------//
/*!
 * Create a tracer from an enumeration.
 */
auto Runner::make_imager(Geometry g) -> SPImager
{
    imager_name_ = std::string{"raytrace_"} + to_cstring(g);
    switch (g)
    {
        CASE_RETURN_FUNC_T(Geometry::orange, make_imager, );
        CASE_RETURN_FUNC_T(Geometry::vecgeom, make_imager, );
        CASE_RETURN_FUNC_T(Geometry::geant4, make_imager, );
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create a tracer of a given type.
 */
template<Geometry G>
auto Runner::make_imager() -> SPImager
{
    using GP = GeoParams_t<G>;

    if constexpr (is_geometry_configured_v<GP>)
    {
        static_assert(is_geometry_configured_v<GP>);
        std::shared_ptr<GP const> geo = this->load_geometry<G>();
        return std::make_shared<RaytraceImager<GP>>(geo);
    }
    else
    {
        CELER_NOT_CONFIGURED(to_cstring(G));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Allocate and perform a raytrace using an enumeration.
 */
auto Runner::make_traced_image(MemSpace m, ImagerInterface& generate_image)
    -> SPImage
{
    switch (m)
    {
        CASE_RETURN_FUNC_T(MemSpace::host, make_traced_image, generate_image);
        CASE_RETURN_FUNC_T(MemSpace::device, make_traced_image, generate_image);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Allocate and perform a raytrace with the given memory/execution space.
 */
template<MemSpace M>
auto Runner::make_traced_image(ImagerInterface& generate_image) -> SPImage
{
    auto image = std::make_shared<Image<M>>(last_image_);

    Stopwatch get_time;
    generate_image(image.get());
    timers_[imager_name_ + '_' + to_cstring(M)] += get_time();

    return image;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
