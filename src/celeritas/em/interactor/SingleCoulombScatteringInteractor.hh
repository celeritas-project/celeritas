members :
    // incident particle data
    G4ParticleDefinition const* particle;
G4double mass;

// constant refs
const G4double cosThetaMax = -1.0;
const G4double cosThetaMin = 1.0;

// CosTheta integration bounds for differential cross section
G4double cosTetMinNuc;
G4double cosTetMaxNuc;

// Ratio of scattering just off electrons vs scattering off (electrons +
// nucleus)
G4double elecRatio;

G4double recoilThreshold;
std::vector<G4double> const* pCuts;
G4double lowEnergyLimit;

// Material data
G4MaterialCutsCouple const* currentCouple;
G4Material const* currentMaterial;
G4int currentMaterialIndex;

void G4eSingleCoulombScatteringModel::SampleSecondaries(
    std::vector<G4DynamicParticle*>* fvect,
    G4MaterialCutsCoupl const* couple,
    G4DynamicParticle const* dp,
    G4double cutEnergy,
    G4double)
{
    G4double kinEnergy = dp->GetKineticEnergy();

    // Mott terminates while WOI absorbs particle
    if (kinEnergy < lowEnergyLimit)
    {
        // return;
        fParticleChange->SetProposedKineticEnergy(0.0);
        fParticleChange->ProposeLocalEnergyDeposit(kinEnergy);
        fParticleChange->ProposeNonIonizingEnergyDeposit(kinEnergy);
        return;
    }

    SetupParticle(dp->GetDefinition());
    DefineMaterial(couple);

    currentElement
        = SelectRandomAtom(couple, particle, kinEnergy, cutEnergy, kinEnergy);

    G4double Z = currentElement->GetZ();
    G4int iz = G4int(Z);
    G4int ia = SelectIsotopeNumber(currentElement);
    // Mott scattering only uses isotope number for the ejected ion, whereas
    // WOI uses the isotope number to determine the mass of the nucleus it
    // actually scatters off. I think it should be the latter for consistency.
    // wokvi->SetTargetMass(G4NucleiProperties::GetNuclearMass(ia, iz));

    // ComputeCrossSectionPerAtom
    G4double cross = Mottcross->GetTotalCross();
    if (cross == 0.0)
        return;

    G4ThreeVector dir = dp->GetMomentumDirection();
    G4ThreeVector newDirection = Mottcross->GetNewDirection();  // wokvi->SampleSingleScattering(cosTetMinNuc,
                                                                // cosThetaMax,
                                                                // elecRatio);
    // G4double cost = newDirection.z();
    newDirection.rotateUz(dir);

    fParticleChange->ProposeMomentumDirection(newDirection);

    // G4double mom2 = wokvi->GetMomentumSquare();
    // G4double trec        = mom2 * (1.0 - cost) / (targetMass + (mass +
    // kinEnergy) * (1.0 - cost)); Mottcross->GetTrec() = mom2 * (1.0 - cost) /
    // (targetMass + mass * mass / targetMass + 2.0 * (kinEnergy + mass));
    G4double trec = Mottcross->GetTrec();
    G4double finalT = kinEnergy - trec;

    if (finalT <= lowEnergyLimit)
    {
        trec = kinEnergy;
        finalT = 0.0;
    }

    fParticleChange->SetProposedKineticEnergy(finalT);
    G4double tcut = recoilThreshold;

    if (pCuts)
    {
        tcut = std::min(tcut, (*pCuts)[currentMaterialIndex]);
        // tcut = std::max(tcut, (*pCuts)[currentMaterialIndex]); <-- WOI uses
        // max?
    }

    if (trec > tcut)
    {
        G4ParticleDefinition* ion = theParticleTable->GetIon(iz, ia, 0.);
        G4double ptot = sqrt(Mottcross->GetMom2Lab());
        G4double plab = sqrt(finalT * (finalT + 2.0 * mass));
        G4ThreeVector p2 = (ptot * dir - plab * newDirection).unit();
        G4DynamicParticle* newdp = new G4DynamicParticle(ion, p2, trec);
        fvect->push_back(newdp);
    }
    else if (trec > 0.)
    {
        fParticleChange->ProposeNonIonizingEnergyDeposit(trec);
        if (trec < tcut)  // isn't this true unless trec == tcut?
            fParticleChange->ProposeLocalEnergyDeposit(trec);
    }
}

void SetupParticle(G4PartcileDefinition const* p)
{
    if (p != particle)
    {
        particle = p;
        mass = particle->GetPDGMass();
        Mottcross->SetupParticle(p);
        // wokvi->SetupParticle(p);
    }
}

void DefineMaterial(G4MaterialCutsCouple const* c)
{
    if (cup != currentCouple)
    {
        currentCouple = cup;
        currentMaterial = cup->GetMaterial();
        currentMaterialIndex = currentCouple->GetIndex();
    }
}

G4double ComputeCrossSectionPerAtom(G4ParticleDefinition const* p,
                                    G4double kinEnergy,
                                    G4double Z,
                                    G4double /*A?*/,
                                    G4double cut,
                                    G4double /*emax?*/)
{
    // Total nuclear + electron cross section
    G4double cross = 0.;
    if (kinEnergy <= 0.)
    {
        return cross;
    }
    // (cosine of) lower theta bound for scattering off nucleus
    cosTetMinNuc = wokvi->SetupKinematic(kinEnergy, currentMaterial);
    // Now have theta integration bounds:
    // Nuclear: [ThetaMinNuc, ThetaMaxNuc]
    // Electron: [ThetaMinNuc, ThetaMax]
    // with the ordering: ThetaMinNuc < ThetaMaxNuc < ThetaMax
    if (cosThetaMax < cosTetMinNuc)
    {
        G4int iz = G4int(Z);
        cosTetMinNuc = wokvi->SetupTarget(iz, cutEnergy);
        cosTetMaxNuc = cosThetaMax;
        if (iz == 1 && cosTetMaxNuc < 0.0 && particle == theProton)
        {
            cosTetMaxNuc = 0.0;
        }
        cross = wokvi->ComputeNuclearCrossSection(cosTetMinNuc, cosTetMaxNuc);
        elecRatio
            = wokvi->ComputeElectronCrossSection(cosTetMinNuc, cosThetaMax);
        cross += elecRatio;
        if (cross > 0.0)
        {
            elecRatio /= cross;
        }
    }
    return cross;
}

// wokvi / mottcross
void SetupParticle(G4ParticleDefinition const* p);
void SetTargetMass(G4double targetMass);
