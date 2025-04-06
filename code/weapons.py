# weapons.py
from settings import *
from math import sin, cos, radians, degrees, atan2, sqrt
from random import uniform, random
from sprites import Laser

class Weapon:
    """Base weapon class for the game"""
    def __init__(self, game):
        self.game = game
        self.name = "Standard Laser"
        self.description = "Basic single shot laser"
        self.cooldown = 0.35       # Time between shots
        self.current_cooldown = 0  # Current cooldown timer
        self.unlocked = True
        
    def update(self, dt):
        """Update weapon cooldown"""
        if self.current_cooldown > 0:
            self.current_cooldown -= dt
    
    def can_fire(self):
        """Check if weapon can fire"""
        return self.current_cooldown <= 0 and self.unlocked
    
    def fire(self, pos, direction):
        """Fire the weapon"""
        if not self.can_fire():
            return False
            
        # Play sound
        play_sound(self.game.audio['laser'])
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Create a single projectile
        self.game.lasers.append(self.game.create_laser(pos, direction))
        
        return True


class TripleShot(Weapon):
    """Fires three projectiles in a spread pattern"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Triple Shot"
        self.description = "Fires 3 projectiles in a spread pattern"
        self.cooldown = 0.6
        self.spread_angle = 25  # Good spread angle
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Play sound
        play_sound(self.game.audio['laser'])
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Get the player's rotation angle directly
        player_rotation = self.game.player.rotation
        base_angle = player_rotation
        
        # Calculate spread positions
        center_rad = radians(base_angle)
        left_rad = radians(base_angle - self.spread_angle)
        right_rad = radians(base_angle + self.spread_angle)
        
        # Add a slight offset to starting positions
        offset = 10  # Pixels to offset each projectile
        
        # Create center projectile
        center_dir = Vector2(sin(center_rad), -cos(center_rad))
        self.game.lasers.append(Laser(self.game.assets['laser'], pos, center_dir))
        
        # Create left projectile with offset
        left_dir = Vector2(sin(left_rad), -cos(left_rad))
        left_offset = Vector2(sin(left_rad) * offset, -cos(left_rad) * offset)
        left_pos = Vector2(pos.x + left_offset.x, pos.y + left_offset.y)
        self.game.lasers.append(Laser(self.game.assets['laser'], left_pos, left_dir))
        
        # Create right projectile with offset
        right_dir = Vector2(sin(right_rad), -cos(right_rad))
        right_offset = Vector2(sin(right_rad) * offset, -cos(right_rad) * offset)
        right_pos = Vector2(pos.x + right_offset.x, pos.y + right_offset.y)
        self.game.lasers.append(Laser(self.game.assets['laser'], right_pos, right_dir))
        
        return True



class RapidFire(Weapon):
    """Fires quickly but with slightly lower damage"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Rapid Fire"
        self.description = "High rate of fire laser"
        self.cooldown = 0.2  # Much faster firing rate
        self.unlocked = False  # Starts locked


class EnemyWeapon:
    """Base weapon class for enemies"""
    def __init__(self, game):
        self.game = game
        self.name = "Standard Enemy Laser"
        self.cooldown = 1.0  # Default cooldown
        self.current_cooldown = 0
        self.damage = 1  # Standard damage
        
    def update(self, dt):
        """Update weapon cooldown"""
        if self.current_cooldown > 0:
            self.current_cooldown -= dt
            
    def can_fire(self):
        """Check if weapon can fire"""
        return self.current_cooldown <= 0
        
    def fire(self, pos, direction):
        """Fire the weapon"""
        if not self.can_fire():
            return False
        
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Create a single projectile
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos, 
            direction
        )
        return True


class TankCannon(EnemyWeapon):
    """Heavy cannon for Tank enemies - slow but powerful"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Tank Cannon"
        self.cooldown = 2.0  # Very slow firing rate
        self.damage = 2  # Double damage
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Tank shots are larger if possible (you could add a scale parameter to your laser)
        # For now we'll just use the regular asset
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos,
            direction
        )
        
        # Play a stronger sound if available
        play_sound(self.game.audio['enemy_laser'])
        
        return True


class SwarmBlaster(EnemyWeapon):
    """Fast, light weapon for Swarm enemies"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Swarm Blaster"
        self.cooldown = 0.5  # Very fast firing
        self.damage = 0.5  # Lower damage
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Add some randomness to direction (inaccurate)
        spread = 0.1  # Amount of random spread
        random_dir = Vector2(
            direction.x + uniform(-spread, spread),
            direction.y + uniform(-spread, spread)
        )
        
        # Normalize the direction vector
        length = sqrt(random_dir.x**2 + random_dir.y**2)
        if length > 0:
            random_dir.x /= length
            random_dir.y /= length
        
        # Fire with slight randomness to direction
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos,
            random_dir
        )
        
        return True


class SniperRifle(EnemyWeapon):
    """Precision long-range weapon for Sniper enemies"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Sniper Rifle"
        self.cooldown = 3.0  # Very slow firing rate
        self.damage = 2  # High damage
        self.projectile_speed = 1.5  # 50% faster projectiles
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Sniper shots are perfectly accurate and faster
        # Create a special laser with higher speed (would need EnemyLaser modifications)
        self.game.shoot_enemy_laser(
            self.game.assets['enemy_laser'], 
            pos,
            direction
        )
        
        # Play a distinct sound if available
        play_sound(self.game.audio['enemy_laser'])
        
        return True


class BomberLauncher(EnemyWeapon):
    """Area-effect weapon for Bomber enemies"""
    def __init__(self, game):
        super().__init__(game)
        self.name = "Bomber Launcher"
        self.cooldown = 2.5  # Moderate firing rate
        self.damage = 1  # Standard damage
        self.spread_count = 5  # Number of projectiles in spread
        
    def fire(self, pos, direction):
        if not self.can_fire():
            return False
            
        # Reset cooldown
        self.current_cooldown = self.cooldown
        
        # Fire multiple projectiles in a spread pattern
        spread_angle = 50  # Total spread in degrees
        angle_step = spread_angle / (self.spread_count - 1)
        
        # Calculate base angle from direction vector
        base_angle = degrees(atan2(direction.y, direction.x))
        
        # Fire spread pattern
        for i in range(self.spread_count):
            shot_angle = base_angle - (spread_angle/2) + (angle_step * i)
            shot_rad = radians(shot_angle)
            spread_dir = Vector2(cos(shot_rad), sin(shot_rad))
            
            self.game.shoot_enemy_laser(
                self.game.assets['enemy_laser'], 
                pos,
                spread_dir
            )
        
        # Play special sound if available
        play_sound(self.game.audio['enemy_laser'])
        
        return True