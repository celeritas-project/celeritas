#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
namespace celeritas
{
namespace optical
{

struct MieMaterialData
{
    real_type forward_g{};  //!< Henyey–Greenstein "g" parameter for forward
                            //!< scattering
    real_type backward_g{};  //!< Henyey–Greenstein "g" parameter for backward
                             //!< scattering
    real_type forward_ratio{};  //!< Fraction of forward vs backward scattering

    //! Whether all data are assigned and valid
    explicit operator bool() const
    {
        return forward_ratio >= 0 && forward_ratio <= 1 && forward_g >= -1
               && forward_g <= 1 && backward_g >= -1 && backward_g <= 1;
    }
};
template<Ownership W, MemSpace M>
struct MieData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OptMatId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<MieMaterialData> mie_record;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !mie_record.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    MieData& operator=(MieData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        mie_record = other.mie_record;
        return *this;
    }
};

}  // namespace optical
}  // namespace celeritas
