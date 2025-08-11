#include "iAnt_controller.h"

static CRange<Real> NN_OUTPUT_RANGE(-1.0f, 1.0f);
static CRange<Real> WHEEL_ACTUATION_RANGE(-16.0f, 16.0f);

/*****
 * Initialize most basic variables and objects here. Most of the setup should be done in the Init(...) function instead
 * of here where possible.
 *****/
iAnt_controller::iAnt_controller() :
    compass(NULL),
    motorActuator(NULL),
    proximitySensor(NULL),
    lightSensor(NULL),
    distanceTolerance(0.0),
    searchStepSize(0.0),
    robotForwardSpeed(0.0),
    robotRotationSpeed(0.0),
    RNG(NULL),
    loopFunctions(NULL),
    isHoldingFood(false),
    action_left_speed(0.0),
    action_right_speed(0.0),
    action_lay_pheromone(false)
{}

/*****
 * Initialize the controller via the XML configuration file. ARGoS typically wants objects & variables initialized here
 * instead of in the constructor(s).
 *****/
void iAnt_controller::Init(TConfigurationNode& node) {

    /* Shorter names, please. #This_Is_Not_Java */
    typedef CCI_PositioningSensor            CCI_PS;
    typedef CCI_DifferentialSteeringActuator CCI_DSA;
    typedef CCI_FootBotProximitySensor       CCI_FBPS;

    /* Initialize the robot's actuator and sensor objects. */
    motorActuator   = GetActuator<CCI_DSA>("differential_steering");
    compass         = GetSensor<CCI_PS>   ("positioning");
    proximitySensor = GetSensor<CCI_FBPS> ("footbot_proximity");
    lightSensor     = GetSensor  <CCI_FootBotLightSensor>("footbot_light");

    CVector2 p(GetPosition());
    startPosition = CVector3(p.GetX(), p.GetY(), 0.0);
}

/*****
 * Primary control loop for this controller object. This function will execute the CPFA logic using the CPFA 
 * enumeration flag once per frame.
 *****/
void iAnt_controller::ControlStep() {
    // Apply RL action (already set via SetAction())
    m_fLeftSpeed = action_left_speed;
    m_fRightSpeed = action_right_speed;
    WHEEL_ACTUATION_RANGE.TruncValue(m_fLeftSpeed);
    WHEEL_ACTUATION_RANGE.TruncValue(m_fRightSpeed);

    motorActuator->SetLinearVelocity(m_fLeftSpeed, m_fRightSpeed);

    if(action_lay_pheromone) {
        layPheromone();
    }

    SetHoldingFood();
}

Real iAnt_controller::maxProximity(int *sensorIndex) {
    Real maxProximity = 0;

    for(int i = 0; i < 6; i++) {
        Real proximityValue = proximitySensor->GetReadings().at(sensorIndex[i]).Value;
        if (proximityValue > maxProximity) {
            maxProximity = proximityValue;
        }
    }

    return maxProximity;
}

int iAnt_controller::maxLightIndex(CCI_FootBotLightSensor::TReadings tReadings,
                                   int sensorIndex[], int numIndices) {
    Real maxVal = 0.0;

    int maxIndex = sensorIndex[0];

    for(int i = 0; i < numIndices; i++) {
        if (maxVal > tReadings[sensorIndex[i]].Value){
            maxVal= tReadings[sensorIndex[i]].Value;
            maxIndex = i;
        };
    }
    return maxIndex;
}

/*****
 * After pressing the reset button in the GUI, this controller will be set to default factory settings like at the
 * start of a simulation.
 *****/
void iAnt_controller::Reset() {
    /* Reset all local variables. */
    isHoldingFood       = false;
    action_left_speed   = 0.0f;
    action_right_speed  = 0.0f;
    action_lay_pheromone = false;
}

/*****
 * Check if the iAnt is finding food. This is defined as the iAnt being within
 * the distance tolerance of the position of a food item. If the iAnt has found
 * food then the appropriate boolean flags are triggered.
 *****/
void iAnt_controller::SetHoldingFood() {

    /* Is the iAnt already holding food? */
    if(!IsHoldingFood()) {

        /* No, the iAnt isn't holding food. Check if we have found food at our
           current position and update the food list if we have. */

        vector<CVector2> newFoodList;
        vector<CColor>   newFoodColoringList;
        size_t i = 0, j = 0;

        for(i = 0; i < loopFunctions->FoodList.size(); i++) {
            if((GetPosition() - loopFunctions->FoodList[i]).SquareLength() < loopFunctions->FoodRadiusSquared) {
                /* We found food! Calculate the nearby food density. */
                isHoldingFood = true;
                //SetLocalResourceDensity();
                j = i + 1;
                break;
            } else {
                /* Return this unfound-food position to the list */
                newFoodList.push_back(loopFunctions->FoodList[i]);
                newFoodColoringList.push_back(CColor::BLACK);
            }
        }

        for( ; j < loopFunctions->FoodList.size(); j++) {
            newFoodList.push_back(loopFunctions->FoodList[j]);
            newFoodColoringList.push_back(CColor::BLACK);
        }

        /* We picked up food. Update the food list minus what we picked up. */
        if(IsHoldingFood()) {
            loopFunctions->FoodList = newFoodList;
        }
    }
    /* Drop off food: We are holding food and have reached the nest. */
    else if((GetPosition() - loopFunctions->NestPosition).SquareLength() < loopFunctions->NestRadiusSquared) {
        isHoldingFood = false;
        loopFunctions->foodReturned++;
    }
}

bool iAnt_controller::IsNearFood() {
    for(int i = 0; i < loopFunctions->FoodList.size(); i++) {
        if ((GetPosition() - loopFunctions->FoodList[i]).SquareLength() < loopFunctions->FoodRadiusSquared) {
            return true;
        }
    }
    return false;
}

bool iAnt_controller::IsNearPheromone() {
    for(int i = 0; i < loopFunctions->Pheromones.size(); i++) {
        if (loopFunctions->Pheromones[i].IsActive() && ((GetPosition() - loopFunctions->Pheromones[i].GetLocation()).SquareLength() < loopFunctions->FoodRadiusSquared)) {
            return true;
        }
    }
    return false;
}

void iAnt_controller::layPheromone() {
    Real timeInSeconds = (Real)(loopFunctions->SimTime / loopFunctions->TicksPerSecond);
    if (!IsNearPheromone()) {
        iAnt_pheromone sharedPheromone(GetPosition(), timeInSeconds, loopFunctions->RateOfPheromoneDecay);
        loopFunctions->Pheromones.push_back(sharedPheromone);
    }
    else{
        for(int i = 0; i < loopFunctions->Pheromones.size(); i++) {
            if (loopFunctions->Pheromones[i].IsActive() && ((GetPosition() - loopFunctions->Pheromones[i].GetLocation()).SquareLength() < loopFunctions->FoodRadiusSquared)) {
                loopFunctions->Pheromones[i].Reset(timeInSeconds);
            }
        }
    }
}



/*****
 * Return the robot's 2D position on the arena.
 *****/
CVector2 iAnt_controller::GetPosition() {
    /* The robot's compass sensor gives us a 3D position. */
    CVector3 position3D = compass->GetReading().Position;
    /* Return the 2D position components of the compass sensor reading. */
    return CVector2(position3D.GetX(), position3D.GetY());
}

REGISTER_CONTROLLER(iAnt_controller, "iAnt_controller")

void iAnt_controller::SetAction(Real left_speed, Real right_speed, bool lay_pheromone) {
    action_left_speed = left_speed;
    action_right_speed = right_speed;
    action_lay_pheromone = lay_pheromone;
}

std::vector<Real> iAnt_controller::GetObservation() {
    std::vector<Real> obs;
    obs.reserve(15);

    // Orientation quaternion components
    obs.push_back(compass->GetReading().Orientation.GetW());
    obs.push_back(compass->GetReading().Orientation.GetX());
    obs.push_back(compass->GetReading().Orientation.GetY());
    obs.push_back(compass->GetReading().Orientation.GetZ());

    // Holding food, near food
    obs.push_back(IsHoldingFood() ? 1.0 : 0.0);
    obs.push_back(IsNearFood() ? 1.0 : 0.0);

    int frontIndexes[6] = {21, 22, 23, 0, 1, 2};
    int leftIndexes[6]  = {3, 4, 5, 6, 7, 8};
    int backIndexes[6]  = {9, 10, 11, 12, 13, 14};
    int rightIndexes[6] = {15, 16, 17, 18, 19, 20};

    // Proximity
    obs.push_back(maxProximity(frontIndexes));
    obs.push_back(maxProximity(leftIndexes));
    obs.push_back(maxProximity(backIndexes));
    obs.push_back(maxProximity(rightIndexes));

    // Pheromone presence
    obs.push_back(IsNearPheromone() ? 1.0 : 0.0);

    // Light readings max per quadrant
    const CCI_FootBotLightSensor::TReadings& tReadings = lightSensor->GetReadings();
    int numIndices = 6;
    int frontLight = maxLightIndex(tReadings, frontIndexes, numIndices);
    int leftLight  = maxLightIndex(tReadings, leftIndexes,  numIndices);
    int backLight  = maxLightIndex(tReadings, backIndexes,  numIndices);
    int rightLight = maxLightIndex(tReadings, rightIndexes, numIndices);

    obs.push_back(tReadings[frontLight].Value);
    obs.push_back(tReadings[leftLight].Value);
    obs.push_back(tReadings[backLight].Value);
    obs.push_back(tReadings[rightLight].Value);

    return obs;
}
