//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.test.hh
//---------------------------------------------------------------------------//

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
/*!
 * Input data for testing algorithms on device.
 */
template<Ownership W, MemSpace M>
struct AlgorithmInputData
{
    template<class T>
    using Items = Collection<T, W, M, ThreadId>;

    // Test sincospi
    Items<double> pi_frac;

    // Test algorithms that take two floating point numbers
    Items<double> a;
    Items<double> b;

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        return !pi_frac.empty() && !a.empty() && a.size() == b.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AlgorithmInputData& operator=(AlgorithmInputData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        pi_frac = other.pi_frac;
        a = other.a;
        b = other.b;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Test results.
 */
template<Ownership W, MemSpace M>
struct AlgorithmOutputData
{
    template<class T>
    using Items = Collection<T, W, M, ThreadId>;

    // Test sincospi
    Items<double> sinpi;
    Items<double> cospi;

    // Test fastpow
    Items<double> fastpow;

    // Test hypot
    Items<double> hypot;

    //! True if sizes are consistent and nonzero
    explicit CELER_FUNCTION operator bool() const
    {
        return !sinpi.empty() && sinpi.size() == cospi.size()
               && !fastpow.empty() && !hypot.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    AlgorithmOutputData& operator=(AlgorithmOutputData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        sinpi = other.sinpi;
        cospi = other.cospi;
        fastpow = other.fastpow;
        hypot = other.hypot;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void resize(AlgorithmOutputData<Ownership::value, M>* output,
                   HostCRef<AlgorithmInputData> const input,
                   size_type)
{
    CELER_EXPECT(output);
    CELER_EXPECT(input);

    resize(&output->sinpi, input.pi_frac.size());
    resize(&output->cospi, input.pi_frac.size());

    resize(&output->fastpow, input.a.size());

    resize(&output->hypot, input.a.size());

    CELER_ENSURE(*output);
}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct AlgorithmTestData
{
    using InputRef = DeviceCRef<AlgorithmInputData>;
    using OutputRef = DeviceRef<AlgorithmOutputData>;

    InputRef input;
    OutputRef output;
    size_type num_threads{};
};

//---------------------------------------------------------------------------//
//! Run on device
void alg_test(AlgorithmTestData);

#if !CELER_USE_DEVICE
inline void alg_test(AlgorithmTestData)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
